import argparse
import pathlib
import logging
import json
from collections import defaultdict
from decimal import Decimal

import cv2
import pandas as pd
import imutils
import numpy as np

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

        # Create dataframe for mouse data
        mouse_data = defaultdict(list)
        for l in user_events:
            if 'type' in l and (l['type'] == 'mousemove' or l['type'] == 'mouseclick'):
                for k, v in l.items():
                    mouse_data[k].append(v)
        mouse_data = pd.DataFrame(mouse_data)

        # Get screen size parameter
        display_w = row['Display Width (pixels)']
        display_h = row['Display Height (pixels)']

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

        # Screen (don't even bother, you would have to perform manual alignment, as reported by the WebGazer team)
        # extension = ".mov"
        # if row['Setting'] == 'PC': extension = ".flv"
        # screen_video = p_dir / f"{participant}{extension}"
        # if not screen_video.exists():
        #     # logger.error(f"Participant {participant} does not have screen video: {screen_video}")
        #     continue
        # screen_cap = cv2.VideoCapture(str(screen_video))

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
        # screen_length = int(screen_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # fps = screen_cap.get(cv2.CAP_PROP_FPS)

        # Timestamps to train each data stream
        # video_timestamp = start_timestamp
        # user_events_timestamp = -1
        # user_events_index = 0
        # webcam_cap = None
        # webcam_timestamp = -1
        # webcam_fps = -1

        # Iterating over the webcam videos using the user events
        for session_name, session in webcam_videos.items():
            
            # Load the webcam video
            webcam_cap = cv2.VideoCapture(str(session['file']))
            webcam_fps = webcam_cap.get(cv2.CAP_PROP_FPS)
            webcam_length = int(webcam_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            webcam_timestamp = session['epoch']

            # Get the mouse data for the current session
            session_mouse_data = mouse_data[(mouse_data['epoch'] > webcam_timestamp) & (mouse_data['epoch'] < webcam_timestamp + webcam_length)]
            mouse_data_available = len(session_mouse_data) > 0
            if mouse_data_available:
                mouse_timestamp = session_mouse_data.iloc[0]['epoch']
                mouse_index = 0
                mouse_pos = (-1,-1)

            logger.debug(f"Webcam video: {session_name}, length: {webcam_length}, fps: {webcam_fps}, timestamp: {webcam_timestamp}, mouse data available: {mouse_data_available}")

            # Iterate over the webcam video
            while True:
                ret, frame = webcam_cap.read()
                if not ret:
                    break

                # Create a blank frame for the screen 
                screen = np.ones((display_h, display_w, 3), dtype=np.uint8) * 255

                # Only process mouse if available
                if mouse_data_available:

                    # Draw the mouse data on the screen
                    while mouse_index < len(session_mouse_data) and mouse_timestamp < webcam_timestamp:
                        # Get the mouse data
                        mouse_x = session_mouse_data.iloc[mouse_index]['clientX']
                        mouse_y = session_mouse_data.iloc[mouse_index]['clientY']
                        mouse_timestamp = session_mouse_data.iloc[mouse_index]['epoch']

                        # Draw the mouse data
                        mouse_pos = (int(mouse_x), int(mouse_y))

                        # Update the mouse index
                        mouse_index += 1

                    # Draw the mouse data
                    cv2.circle(screen, mouse_pos, 5, (0, 0, 255), -1)

                # Concat the screen and the webcam
                webcam_h, webcam_w, _ = frame.shape
                pad_h = (display_h - webcam_h) // 2
                pad_w = (display_w - webcam_w) // 2
                frame = cv2.copyMakeBorder(frame, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                frame = np.concatenate((screen, frame), axis=1)

                cv2.imshow('frame', imutils.resize(frame, width=1200))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Update the time
                webcam_timestamp += 1000/webcam_fps

            # Release the webcam video
            webcam_cap.release()
            cv2.destroyAllWindows()

            # break

            # Store the webcam data
            # webcam_data = pd.DataFrame(webcam_data)
            # webcam_data.to_pickle(args.output / f"{participant}_{session_name}_webcam.pkl")

        break

    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extracts the eye tracking data from the webgazer dataset')
    parser.add_argument('-d', '--dataset', required=True, type=str, help='Directory of the WebGazerETRA2018Dataset folder')
    parser.add_argument('-o', '--output', default="./output", type=str, help='Directory to save the extracted data')
    args = parser.parse_args()

    main(args)