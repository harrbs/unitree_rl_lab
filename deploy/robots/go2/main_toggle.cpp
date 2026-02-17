#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "LinearInterpolator.h"
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>

#include <chrono>
#include <thread>

using unitree::robot::b2::MotionSwitcherClient;

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = std::make_shared<Keyboard>();

static bool release_high_level_mode()
{
    MotionSwitcherClient msc;
    msc.SetTimeout(10.0f);
    msc.Init();

    while (true)
    {
        std::string robotForm, motionName;
        int32_t ret = msc.CheckMode(robotForm, motionName);
        if (ret != 0)
        {
            spdlog::warn("CheckMode failed: {}", ret);
            return false;
        }
        if (motionName.empty())
        {
            return true;
        }
        ret = msc.ReleaseMode();
        if (ret != 0)
        {
            spdlog::warn("ReleaseMode failed: {}", ret);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    return true;
}

static void hold_current_pose_once()
{
    auto kp = param::config["FSM"]["FixStand"]["kp"].as<std::vector<float>>();
    auto kd = param::config["FSM"]["FixStand"]["kd"].as<std::vector<float>>();
    const int n = std::min<int>(12, kp.size());

    std::array<float, 12> q_now{};
    {
        std::lock_guard<std::mutex> lock(FSMState::lowstate->mutex_);
        for (int i = 0; i < n; i++)
        {
            q_now[i] = FSMState::lowstate->msg_.motor_state()[i].q();
        }
    }

    FSMState::lowcmd->lock();
    for (int i = 0; i < n; i++)
    {
        auto & motor = FSMState::lowcmd->msg_.motor_cmd()[i];
        motor.mode() = 1;
        motor.q() = q_now[i];
        motor.dq() = 0;
        motor.kp() = kp[i];
        motor.kd() = kd[i];
        motor.tau() = 0;
    }
    FSMState::lowcmd->unlockAndPublish();
}

static void hold_current_pose_until_f()
{
    std::cout << "Holding current pose until 'f' is pressed..." << std::endl;
    while (true)
    {
        if (FSMState::keyboard)
        {
            FSMState::keyboard->update();
            if (FSMState::keyboard->key() == "f")
            {
                std::cout << "Key 'f' received. Exiting hold." << std::endl;
                break;
            }
        }

        hold_current_pose_once();
        std::this_thread::sleep_for(std::chrono::milliseconds(2)); // ~500Hz
    }
}

static void override_fixstand_to_final_only()
{
    auto fix = param::config["FSM"]["FixStand"];
    fix["ts"] = std::vector<float>{0.0f, 1.0f};

    std::vector<std::vector<float>> qs;
    qs.push_back(std::vector<float>{});  // start: current pose
    qs.push_back(std::vector<float>{
        0.0f, 0.8f, -1.5f,
        0.0f, 0.8f, -1.5f,
        0.0f, 0.8f, -1.5f,
        0.0f, 0.8f, -1.5f
    });
    fix["qs"] = qs;
}

static std::unique_ptr<CtrlFSM> build_fsm_start_fixstand()
{
    auto cfg = param::config["FSM"];
    auto fsms = cfg["_"];

    for (auto it = fsms.begin(); it != fsms.end(); ++it)
    {
        std::string fsm_name = it->first.as<std::string>();
        int id = it->second["id"].as<int>();
        FSMStringMap.insert({id, fsm_name});
    }

    std::shared_ptr<BaseState> fixstand_state;
    for (auto it = fsms.begin(); it != fsms.end(); ++it)
    {
        std::string fsm_name = it->first.as<std::string>();
        if (fsm_name != "FixStand") continue;
        int id = it->second["id"].as<int>();
        std::string fsm_type = it->second["type"] ? it->second["type"].as<std::string>() : fsm_name;
        auto fsm_class = getFsmMap().find("State_" + fsm_type);
        if (fsm_class == getFsmMap().end()) {
            throw std::runtime_error("FSM: Unknown FSM type " + fsm_type);
        }
        fixstand_state = fsm_class->second(id, fsm_name);
        break;
    }
    if (!fixstand_state) {
        throw std::runtime_error("FSM: FixStand state not found in config");
    }

    auto fsm = std::make_unique<CtrlFSM>(fixstand_state);
    for (auto it = fsms.begin(); it != fsms.end(); ++it)
    {
        std::string fsm_name = it->first.as<std::string>();
        if (fsm_name == "FixStand") continue;
        int id = it->second["id"].as<int>();
        std::string fsm_type = it->second["type"] ? it->second["type"].as<std::string>() : fsm_name;
        auto fsm_class = getFsmMap().find("State_" + fsm_type);
        if (fsm_class == getFsmMap().end()) {
            throw std::runtime_error("FSM: Unknown FSM type " + fsm_type);
        }
        auto state_instance = fsm_class->second(id, fsm_name);
        fsm->add(state_instance);
    }
    return fsm;
}

static void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::go2::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if(!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

int main(int argc, char** argv)
{
    auto vm = param::helper(argc, argv);

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     Go2 Controller (Keyboard + Toggle) \n";

    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

    init_fsm_state();

    std::cout << "Press Enter to switch high-level -> low-level, then hold current pose..." << std::endl;
    std::cin.ignore();
    std::cout << "Enter received." << std::endl;

    bool switched = release_high_level_mode();
    if (switched)
    {
        std::cout << "High-level -> low-level mode switch complete." << std::endl;
    }
    else
    {
        std::cout << "High-level mode not available or no response. Skipping switch." << std::endl;
    }

    hold_current_pose_until_f();

    override_fixstand_to_final_only();

    auto fsm = build_fsm_start_fixstand();
    fsm->start();

    std::cout << "Keyboard shortcuts:\n";
    std::cout << "  f  -> LT + A (FixStand)\n";
    std::cout << "  g  -> Start (Velocity)\n";
    std::cout << "  h  -> LT + B (Passive)\n";
    std::cout << "  w/a/s/d or arrows -> velocity\n";

    while (true)
    {
        sleep(1);
    }
    return 0;
}
