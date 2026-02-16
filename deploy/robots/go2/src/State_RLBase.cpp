#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

#include <algorithm>
#include <chrono>
#include <mutex>
#include <spdlog/spdlog.h>

namespace {
std::shared_ptr<unitree::robot::go2::subscription::SportModeState> g_sport_state;
std::once_flag g_sport_state_once;
}

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    std::call_once(g_sport_state_once, [](){
        g_sport_state = std::make_shared<unitree::robot::go2::subscription::SportModeState>();
        g_sport_state->wait_for_connection();
    });

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }

    static auto last_print = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    if (now - last_print < std::chrono::milliseconds(200)) {
        return;
    }
    last_print = now;

    float cmd_x = 0.0f;
    float cmd_y = 0.0f;
    float cmd_yaw = 0.0f;
    if (env->robot->data.joystick) {
        const auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];
        cmd_x = std::clamp(env->robot->data.joystick->ly(),
                           cfg["lin_vel_x"][0].as<float>(),
                           cfg["lin_vel_x"][1].as<float>());
        cmd_y = std::clamp(-env->robot->data.joystick->lx(),
                           cfg["lin_vel_y"][0].as<float>(),
                           cfg["lin_vel_y"][1].as<float>());
        cmd_yaw = std::clamp(-env->robot->data.joystick->rx(),
                             cfg["ang_vel_z"][0].as<float>(),
                             cfg["ang_vel_z"][1].as<float>());
    }

    if (g_sport_state && !g_sport_state->isTimeout()) {
        Eigen::Vector3f vel = Eigen::Vector3f::Zero();
        float yaw_speed = 0.0f;
        {
            std::lock_guard<std::mutex> lock(g_sport_state->mutex_);
            vel = g_sport_state->velocity();
            yaw_speed = g_sport_state->msg_.yaw_speed();
        }
        spdlog::info("[CMD] x:{:.2f} y:{:.2f} yaw:{:.2f} | [VEL] x:{:.2f} y:{:.2f} z:{:.2f} yaw:{:.2f}",
                     cmd_x, cmd_y, cmd_yaw, vel.x(), vel.y(), vel.z(), yaw_speed);
    } else {
        spdlog::info("[CMD] x:{:.2f} y:{:.2f} yaw:{:.2f} | [VEL] N/A (sportmodestate timeout)",
                     cmd_x, cmd_y, cmd_yaw);
    }
}
