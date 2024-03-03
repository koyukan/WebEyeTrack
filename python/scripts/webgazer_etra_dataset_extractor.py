import argparse
import pathlib
import logging
import json
from collections import defaultdict
from decimal import Decimal

import cv2
import pandas as pd
import imutils

from models import WebCamVideo

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def main(args):
    dataset = pathlib.Path(args.dataset)

    # Load the meta file
    meta = pd.read_csv(f"{dataset}/participant_characteristics.csv")
    
    # Iterate over all the participants data
    for idx, row in meta.iterrows():
        participant = row['Participant ID']

        # Load the data
        p_dir = dataset / participant
        if not p_dir.exists():
            logger.warn(f"Participant {participant} does not exist")
            continue

        # Load the data (user events, webcam, screen, and tobii)
        logger.debug(f"Loading participant {participant} data")

        # User events (its a json file with its stem being the timestamp)
        user_events = list(p_dir.glob("*.json"))
        if len(user_events) == 0:
            logger.error(f"Participant {participant} does not have user events")
            continue
        user_events = user_events[0]
        start_timestamp = int(user_events.stem)
        logger.debug(f"Start timestamp: {start_timestamp}")

        # Parse the user events to get window parameters
        with open(user_events) as f:
            user_events = json.load(f)

        for l in user_events:
            if 'windowX' in l and l['windowX'] != None:
                wg_window_x = int(l['windowX'])
                wg_window_y = int(l['windowY'])
                wg_window_inner_width = int(l['windowInnerWidth'])
                wg_window_inner_height = int(l['windowInnerHeight'])
                wg_window_outer_width = int(l['windowOuterWidth'])
                wg_window_outer_height = int(l['windowOuterHeight'])
                break

        # Webcam
        webcam_videos = {}
        for l in user_events:
            if 'type' in l and l['type'] == 'recording start':
                session_id = l['sessionId']
                file = session_id.replace("/", '-')
        
                # Filter the webcam videos to only include the ones we want (fitts and writing activites)
                # if file.find('_writing') == -1 and file.find('dot_test.') == -1 and file.find('dot_test_final.') == -1:
                #     continue

                # Else store the webcam video
                filepath = p_dir / f"{file}.webm"
                session_name = session_id.split('/')[-1]
                epoch = l['epoch']
                webcam_videos[session_name] = {
                    "file": filepath,
                    "name": session_name, 
                    "epoch": epoch
                }
        # logger.debug(f"Webcam videos: {webcam_videos}")

        # Sort the webcam videos by epoch
        webcam_videos = dict(sorted(webcam_videos.items(), key=lambda item: item[1]['epoch']))

        # Screen
        extension = ".mov"
        if row['Setting'] == 'PC': extension = ".flv"
        screen_video = p_dir / f"{participant}{extension}"
        if not screen_video.exists():
            # logger.error(f"Participant {participant} does not have screen video: {screen_video}")
            continue
        screen_cap = cv2.VideoCapture(str(screen_video))

        # Tobii
        tobii_log = p_dir / f"{participant}.txt"
        if not tobii_log.exists():
            logger.error(f"Participant {participant} does not have tobii log: {tobii_log}")
            continue

        tobii_data = defaultdict(list)
        with open(tobii_log, 'r') as f:
            for line in f:
                l = json.loads(line, parse_float=Decimal)
                rsgx = float(l['right_gaze_point_on_display_area'][0])
                rsgy = float(l['right_gaze_point_on_display_area'][1])
                lsgx = float(l['left_gaze_point_on_display_area'][0])
                lsgy = float(l['left_gaze_point_on_display_area'][1])
                timestamp = round( l['true_time'] * 1000 )
                rpv = l['right_pupil_validity']
                lpv = l['left_pupil_validity']
                tobii_data['timestamp'].append(timestamp)
                tobii_data['rpv'].append(rpv)
                tobii_data['lpv'].append(lpv)
                tobii_data['rsgx'].append(rsgx)
                tobii_data['rsgy'].append(rsgy)
                tobii_data['lsgx'].append(lsgx)
                tobii_data['lsgy'].append(lsgy)

        tobii_data = pd.DataFrame(tobii_data)

        # Iterate over the screen capture video
        screen_length = int(screen_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = screen_cap.get(cv2.CAP_PROP_FPS)

        # Timestamps to train each data stream
        video_timestamp = start_timestamp
        user_events_timestamp = -1
        user_events_index = 0
        webcam_cap = None
        webcam_timestamp = -1
        webcam_fps = -1

        # Loop over the video
        for i in range(screen_length):

            # Update the video timestamp
            video_timestamp += 1000 / fps

            # Load frame
            ret, frame = screen_cap.read()
            if not ret:
                logger.error(f"Could not read frame {i} from screen video")
                break

            # Iterate over the event data
            while user_events_timestamp < video_timestamp:
                if user_events_index >= len(user_events):
                    break

                user_events_timestamp = user_events[user_events_index]['epoch']
                user_events_index += 1

                latest_user_event = user_events[user_events_index - 1]
                if 'type' in latest_user_event:
                    if latest_user_event['type'] == 'recording start':
                        if webcam_cap is not None:
                            webcam_cap.release()
                        
                        session_id = latest_user_event['sessionId']
                        session_name = session_id.split('/')[-1]
                        if not session_name in webcam_videos:
                            continue

                        webcam_video = webcam_videos[session_name]
                        # print(webcam_videos, session_name, webcam_video)
                        webcam_cap = cv2.VideoCapture(str(webcam_video['file']))
                        webcam_timestamp = webcam_video['epoch']
                        webcam_fps = webcam_cap.get(cv2.CAP_PROP_FPS)

            # Align webcam with screen
            # import pdb; pdb.set_trace()
            webcam_frame = None
            if webcam_cap is not None:
                while webcam_timestamp < video_timestamp:
                    ret, webcam_frame = webcam_cap.read()
                    if not ret:
                        logger.error(f"Could not read frame {i} from webcam video")
                        break
                    webcam_timestamp += 1000 / webcam_fps

            # Process the frame
            # import pdb; pdb.set_trace()
            h, w = frame.shape[:2]

            # Draw informatics of the video (timestamp, etc) with black background
            cv2.rectangle(frame, (0, 0), (w, 50), (255, 255, 255), -1)
            cv2.putText(frame, f"Participant: {participant}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, f"Timestamp: {video_timestamp}", (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, cv2.LINE_AA)

            # Concatenate the frames
            if type(webcam_frame) is not type(None):
                frame = cv2.hconcat([frame, webcam_frame])

            # Show the frame
            cv2.imshow('frame', imutils.resize(frame, width=800))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        break

    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extracts the eye tracking data from the webgazer dataset')
    parser.add_argument('-d', '--dataset', required=True, type=str, help='Directory of the WebGazerETRA2018Dataset folder')
    parser.add_argument('-o', '--output', default="./output", type=str, help='Directory to save the extracted data')
    args = parser.parse_args()

    main(args)