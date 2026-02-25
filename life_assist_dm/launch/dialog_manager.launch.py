import os

from launch import LaunchDescription
from launch.actions import TimerAction, SetEnvironmentVariable
from launch_ros.actions import Node

def generate_launch_description():
    # stt_node = Node(
    #     package='life_assist_dm',
    #     executable='stt_node',
    #     name='stt_node',
    #     output='screen',
    #     parameters=[{
    #         'whisper_model': 'base',
    #         'gpt_model': 'gpt-4o-mini-2024-07-18',
    #         'duration': 3,
    #         'call_sign': ['로봇', 'robot', '로보', '로그', '로부'],
    #     }]
    # )

    # tts_node = Node(
    #     package='life_assist_dm',
    #     executable='tts_node',
    #     name='tts_node',
    #     output='screen',
    #     parameters=[{

    #     }]
    # )

    # dialog_manager_node = TimerAction(
    #     period=2.0,  # seconds
    #     actions=[
    #         Node(
    #             package='life_assist_dm',
    #             executable='dialog_manager',
    #             name='dialog_manager',
    #             output='screen',
    #             parameters=[{
    #                 'service_list': ['cognitive', 'emotional', 'physical'],
    #                 'gpt_model': 'gpt-4o-mini-2024-07-18',
    #                 'user_config': 'user1.csv',
    #             }]
    #         )
    #     ]
    # )
    
    dialog_manager_node = Node(
                package='life_assist_dm',
                executable='dialog_manager',
                name='dialog_manager',
                output='screen',
                arguments=['--ros-args', '--log-level', 'warn'],  # rqt 메모리 과부하 방지: 로그 레벨 WARN으로 설정
                parameters=[{
                    'service_list': ['cognitive', 'emotional', 'physical'],
                    'gpt_model': 'gpt-4o-mini-2024-07-18',
                    'user_config': 'user1.csv',
                    'preset_user_name': '권서연',  # 미리 지정된 사용자 이름 (있으면 이름 묻는 단계 건너뜀)
        }]
    )

    return LaunchDescription([
        # UTF-8 인코딩 설정 (rqt_service_caller 한글 처리 오류 방지)
        SetEnvironmentVariable('LC_ALL', 'C.UTF-8'),
        SetEnvironmentVariable('LANG', 'C.UTF-8'),
        SetEnvironmentVariable('PYTHONIOENCODING', 'utf-8'),
        # stt_node,
        # tts_node,
        dialog_manager_node  # will start after 2 seconds
    ])
