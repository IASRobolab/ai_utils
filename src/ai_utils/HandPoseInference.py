import mediapipe as mp
import cv2


class HandPoseInference:

    def __init__(self, display=False, static_image_mode=False, model_complexity=1, max_num_hands=2,
                 min_detection_confidence=0.3, min_tracking_confidence=0.3, flip_image=True, flatten=True):

        self.flip_image = flip_image
        self.display = display
        self.flatten = flatten

        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(static_image_mode=static_image_mode, model_complexity=model_complexity,
                                    max_num_hands=max_num_hands, min_detection_confidence=min_detection_confidence,
                                    min_tracking_confidence=min_tracking_confidence)

    def get_hand_pose(self, img):
        '''
        This function returns a list cointaining hand world points (with coordinate frame centered in the hand center)
        :param img: the image on which to search for hands
        :return: the image with hands points on it if self.diplay = true or the original image otherwise AND
        a dictionary with at most two keys (left and right) containing a vstisble length list of hands
        depending on detection and max_num_hands param used in initialization
        '''
        if self.flip_image:
            img = cv2.flip(img, 1)

        img.flags.writeable = False
        # Convert the BGR image to RGB before processing.
        results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img.flags.writeable = True

        if self.flip_image:
            img = cv2.flip(img, 1)

        hands_detected = {}
        # if not hands found on image return None
        if not results.multi_hand_landmarks:
            hands_detected = None
        else:
            if self.display:
                if self.flip_image:
                    img = cv2.flip(img, 1)
                image_height, image_width, _ = img.shape
                for hand_landmarks in results.multi_hand_landmarks:
                    for point_handmark in hand_landmarks.landmark:
                        point_position_x = int(point_handmark.x * image_width)
                        point_position_y = int(point_handmark.y * image_height)
                        # TODO: use z position to increase and decrease circle size
                        # print(point_position_x, point_position_y)
                        cv2.circle(img, (point_position_x, point_position_y), 3, (0, 0, 255), 3)
                if self.flip_image:
                    img = cv2.flip(img, 1)

                cv2.namedWindow("HandPose", cv2.WINDOW_NORMAL)
                cv2.imshow("HandPose", img)

                if cv2.waitKey(1) == ord('q'):
                    print("Closed HandPose Image Viewer.")
                    exit(0)

            # retrieve hand world coordinate points
            for hand_idx in range(len(results.multi_hand_world_landmarks)):
                hand_output = []
                hand_landmarks = results.multi_hand_world_landmarks[hand_idx]
                hand_type = results.multi_handedness[hand_idx].classification[0].label
                for point_handmark in hand_landmarks.landmark:
                    point_x = point_handmark.x
                    point_y = point_handmark.y
                    point_z = point_handmark.z
                    hand_output.append([point_x, point_y, point_z])
                if self.flatten:
                    hand_output = [item for sublist in hand_output for item in sublist]
                if not (hand_type in hands_detected.keys()):
                    hands_detected[hand_type] = []
                hands_detected[hand_type] = hand_output

        return hands_detected
