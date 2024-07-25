#ifndef _FSM_HPP_
#define _FSM_HPP_
#include <ros/ros.h>
#include <ros/time.h>
class CtrlFSM
{
public:
    enum State_t
    {
        INIT_PARAM = 1,
        TAKEOFF,
        HOVER,
        RUNNING,
        LANDING
    };
    State_t now_state = INIT_PARAM;
    State_t last_state = INIT_PARAM;
    ros::Time last_try_offboard_time;
    ros::Time last_try_arm_time;
    ros::Time last_try_land_time;
    void Init_FSM()
    {
        now_state = INIT_PARAM;
        last_state = INIT_PARAM;
        offboard_flag = false;
        takeoff_over_flag = false;
        land_flag = false;
        cmd_vaild_flag = false;
        arm_done_flag = false;

#define TIME_OFFSET_SEC 1000
        last_recv_pva_time = ros::Time::now() - ros::Duration(TIME_OFFSET_SEC);
        last_try_offboard_time = ros::Time::now() - ros::Duration(TIME_OFFSET_SEC);
        last_try_arm_time = ros::Time::now() - ros::Duration(4);
        last_try_land_time = ros::Time::now() - ros::Duration(TIME_OFFSET_SEC);
        ROS_INFO("init FSM");
    }

    inline void process()
    {
        last_state = now_state;
        switch (now_state)
        {
        case INIT_PARAM:
        {
            if (get_arm_flag() && get_offboard_flag())
            {
                now_state = TAKEOFF;
            }
            break;
        }
        case TAKEOFF:
        {
            if (get_takeoff_over_flag())
            {
                now_state = HOVER;
            }
            break;
        }
        case HOVER:
        {
            if (is_cmd_vaild())
            {
                now_state = RUNNING;
            }

            if (get_land_flag())
            {
                now_state = LANDING;
            }

            break;
        }

        case RUNNING:
        {
            if (!is_cmd_vaild())
            {
                now_state = HOVER;
            }

            if (get_land_flag())
            {
                now_state = LANDING;
            }
            break;
        }

        default:
            break;
        }
    }
    // ********************************
    bool get_offboard_flag()
    {
        return offboard_flag;
    }
    void set_offboard_flag(bool flag)
    {
        offboard_flag = flag;
    }

    bool get_takeoff_over_flag()
    {
        return takeoff_over_flag;
    }
    void set_takeoff_over_flag(bool flag)
    {
        takeoff_over_flag = flag;
    }

    bool get_arm_flag()
    {
        return arm_done_flag;
    }
    void set_arm_flag(bool flag)
    {
        arm_done_flag = flag;
    }

    bool get_land_flag()
    {
        return land_flag;
    }

    void set_land_flag(bool flag)
    {
        land_flag = flag;
    }
    // ********************************

    void update_cmd_update_time(ros::Time now_time)
    {
        last_recv_pva_time = now_time;
    }
    bool is_cmd_vaild()
    {

        if (ros::Time::now() - last_recv_pva_time < ros::Duration(1.0))
        {
            return true;
        }
        else
        {
            if(static_cast<int>((ros::Time::now() - last_recv_pva_time).toSec())/3 == 0)
            {
                ROS_INFO_STREAM("cmd timeout(s) "<<ros::Time::now() - last_recv_pva_time);
            }
            return false;
        }
    }

private:
    bool offboard_flag;
    bool takeoff_over_flag;
    bool cmd_vaild_flag;
    bool arm_done_flag;
    bool land_flag;
    ros::Time last_recv_pva_time;

    // add takeoff cmd recv flag;
};

#endif