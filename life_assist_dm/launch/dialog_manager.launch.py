import os

from launch import LaunchDescription
from launch.actions import TimerAction, SetEnvironmentVariable
from launch_ros.actions import Node

def generate_launch_description():
    dialog_manager_node = Node(
                package='life_assist_dm',
                executable='dialog_manager',
                name='dialog_manager',
                output='screen',
                arguments=['--ros-args', '--log-level', 'warn'],  # rqt 메모리 과부하 방지: 로그 레벨 WARN으로 설정
                parameters=[{
                    'service_list': ['cognitive', 'emotional', 'physical'],
                    'gpt_model': 'gpt-4o-mini-2024-07-18',
                    'user_config': '권서연.xlsx',
        }]
    )

    return LaunchDescription([
        # UTF-8 인코딩 설정 (rqt_service_caller 한글 처리 오류 방지)
        SetEnvironmentVariable('LC_ALL', 'C.UTF-8'),
        SetEnvironmentVariable('LANG', 'C.UTF-8'),
        SetEnvironmentVariable('PYTHONIOENCODING', 'utf-8'),
        dialog_manager_node  # will start after 2 seconds
    ])
