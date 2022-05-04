import cv2
import numpy as np
import torch
from torch.nn import functional as F
import sys
from mmt import models
from mmt.utils.serialization import load_checkpoint, copy_state_dict
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

import pdb


class FastBaseTransform(torch.nn.Module):
    """
    Transform that does all operations on the GPU for super speed.
    This doesn't support a lot of config settings and should only be used for production.
    Maintain this as necessary.
    """
    MEANS = (103.94, 116.78, 123.68)
    STD = (57.38, 57.12, 58.40)

    def __init__(self):
        super().__init__()
        self.mean = torch.Tensor(self.MEANS).float().cuda()[None, :, None, None]
        self.std = torch.Tensor(self.STD).float().cuda()[None, :, None, None]

    def forward(self, img):
        self.mean = self.mean.to(img.device)
        self.std = self.std.to(img.device)
        # img assumed to be a pytorch BGR image with channel order [n, h, w, c]
        img_size = (256, 128)
        img = img.permute(0, 3, 1, 2).contiguous()
        img = F.interpolate(img, img_size, mode='bilinear', align_corners=False)
        img = (img - self.mean) / self.std
        img = img[:, (2, 1, 0), :, :].contiguous()
        # Return value is in channel order [n, c, h, w] and RGB
        return img


class Reidentificator:
    '''
    Reidentify an object on an image using the inference output of another AI algorithm and calibration
    '''

    def __init__(self, class_target, model_weights, display_img=False):
        '''
        initialize the Re-identificator object
        :param class_target: the class of the object you want to track
        :param display_img: boolean value to return an image which create a bounding box around the
        re-identified object
        '''

        self.class_target = class_target
        self.display_img = display_img

        self.transform = torch.nn.DataParallel(FastBaseTransform()).cuda()
        print('Loading REID model...', end='')
        self.model_REID = models.create('resnet_ibn50a', pretrained=False,
                                        num_features=256, dropout=0,
                                        num_classes=0, norm=True)
        self.model_REID.cuda()
        self.model_REID = torch.nn.DataParallel(self.model_REID)
        try:
            checkpoint = load_checkpoint(model_weights)
        except ValueError:
            print('\n\033[91mWeights not found in ' + model_weights + ". You must download them "
                                                                      "in that directory.\033[0m")
            exit(1)
        copy_state_dict(checkpoint['state_dict'], self.model_REID)
        self.model_REID.eval()
        print('Done.')

        self.iteration_number = 0
        self.required_calibration_measures = 300
        self.std_dev_confidence = 2.2
        self.calibrated = False
        self.calibration_finished = False

        self.feats = []
        self.feats_distances = []

        self.feature_threshold = float
        self.mahalanobis_deviation_const = float
        self.std_pers = float
        self.mean_pers = float

    def calibrate_person(self, rgb, inference_output):
        '''
        Function used to calibrate the reidentificator with the object image. This function should be called iteratively
        until it returns True (i.e., when the object is calibrated)
        :param rgb: the image in which there is the object
        :param inference_output: a dictionary containing the inferences obtained by an instance segmentation algorithm
        (e.g., Yolact++)
        :return: A boolean which confirm if the object has been correctly calibrated or not
        '''
        try:
            boxes = inference_output[self.class_target]['boxes']
            masks = inference_output[self.class_target]['masks']
        except KeyError:
            return self.calibrated

        if len(boxes) > 1:
            print('WARNING: MORE THAN ONE PERSON DETECTED DURING CALIBRATION!')
            self.iteration_number = 0
            self.feats = []
        else:
            for i in range(3):
                rgb[:, :, i] = rgb[:, :, i] * masks[0]
            percentage = self.iteration_number / self.required_calibration_measures * 100
            if percentage % 10 == 0:
                print("CALIBRATING ", int(percentage), "%")
            img_person = self.transform(
                torch.from_numpy(rgb[boxes[0][1]:boxes[0][3], boxes[0][0]:boxes[0][2], :]).unsqueeze(0).cuda().float())
            self.iteration_number += 1
            self.feats.append(self.model_REID(img_person.cuda()).data.cpu()[0].numpy())

            # after meas_init measures we terminate the initialization
            if self.iteration_number >= self.required_calibration_measures:
                print('\nCALIBRATION FINISHED')

                self.calibrated = True
                # shuffle features
                self.feats = shuffle(self.feats)
                calibration_threshold_slice = int(len(self.feats)*2/3)
                # compute mean, std and mahalanobis
                feat_calibrate = self.feats[:calibration_threshold_slice]
                self.mean_pers = np.mean(np.array(feat_calibrate), axis=0)
                self.std_pers = np.std(np.array(feat_calibrate), axis=0)
                self.mahalanobis_deviation_const = np.sqrt(self.mean_pers.shape[0])
                # compute threshold
                feat_threshold = self.feats[calibration_threshold_slice:]
                person_mean_feat = np.tile(self.mean_pers, (len(feat_threshold), 1))
                person_std_feat = np.tile(self.std_pers, (len(feat_threshold), 1))
                dist_threshold = np.linalg.norm((feat_threshold - person_mean_feat) / (self.mahalanobis_deviation_const * person_std_feat), axis=1)

                # plt.plot(np.arange(len(dist_threshold)), np.array(dist_threshold))
                # plt.plot(np.ones(len(dist_threshold)) * np.mean(np.array(dist_threshold)))
                # plt.plot(np.ones(len(dist_threshold)) * (np.mean(np.array(dist_threshold)) + self.std_dev_confidence * np.std(np.array(dist_threshold))))
                # plt.legend(["Thresholds", "Mean", "2.2 std"])
                # plt.grid(True)
                # plt.show()

                self.feature_threshold = np.mean(np.array(dist_threshold)) + self.std_dev_confidence * np.std(np.array(dist_threshold))
                print("\nTHRESHOLD: %.4f" % self.feature_threshold)

        return self.calibrated

    def reidentify(self, rgb, inference_output):
        '''
        Used to reidentify the calibrated object on the image (if present)
        :param rgb: the image in which there should be the object to reidentify
        :param inference_output: a dictionary containing the inferences obtained by an instance segmentation algorithm
        (e.g., Yolact++)
        :return: the mask of the targeted object if reidentified
        '''
        rgb = rgb.copy()
        if not self.calibrated:
            sys.exit("Error: Reidentificator not calibrated!")

        try:
            boxes = inference_output[self.class_target]['boxes']
            masks = inference_output[self.class_target]['masks']

            img_persons = []
            # COPY THE FEATURES TEMPLATE ACCORDINGLY TO NUMBER OF DETECTED PERSONS FOR FAST DISTANCE COMPUTATION
            person_mean_feat = np.tile(self.mean_pers, (len(boxes), 1))
            person_std_feat = np.tile(self.std_pers, (len(boxes), 1))
            # CUT THE BOUNDING BOXES OF THE DETECTED PERSONS OVER THE IMAGE
            for id in range(len(boxes)):
                rgb_new = rgb.copy()
                for i in range(3):
                    rgb_new[:, :, i] = rgb_new[:, :, i] * masks[id]
                person_bb = self.transform(
                    torch.from_numpy(rgb_new[boxes[id][1]:boxes[id][3], boxes[id][0]:boxes[id][2], :]).unsqueeze(
                        0).cuda().float())
                img_persons.append(person_bb[0])
            img_persons = [img_person.cuda().float() for img_person in img_persons]
            img_persons = torch.stack(img_persons, 0)
            # PASS THE IMAGES INSIDE THE EXTERNAL NETWORK
            feat_pers = self.model_REID(img_persons).data.cpu()
            # COMPUTE FEATURES DISTANCES
            dist = np.linalg.norm((feat_pers - person_mean_feat) / (self.mahalanobis_deviation_const * person_std_feat),
                                  axis=1)
            # print(dist)
            # RETURN REIDENTIFIED CLASS IFF THE DISTANCE BETWEEN FEATURE IS NO MORE THAN A CALIBRATED THRESHOLD
            if np.min(dist) > self.feature_threshold:
                reidentified_class = None
            else:
                target_idx = np.argmin(dist)
                reidentified_class = {"class": self.class_target,
                                      "boxes": boxes[target_idx],
                                      "masks": masks[target_idx]}

                # DISPLAY REIDENTIFIED IMAGE
                if self.display_img:
                    x1, y1, x2, y2 = boxes[target_idx]
                    cv2.rectangle(rgb, (x1, y1), (x2, y2), (255, 255, 255), 5)

                    text_str = 'TARGET'

                    font_face = cv2.FONT_HERSHEY_DUPLEX
                    font_scale = 0.6
                    font_thickness = 1

                    text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                    text_pt = (x1, y1 - 3)
                    text_color = [255, 255, 255]

                    cv2.rectangle(rgb, (x1, y1), (x1 + text_w, y1 - text_h - 4), (0, 0, 0), -1)
                    cv2.putText(rgb, text_str, text_pt, font_face, font_scale, text_color, font_thickness,
                                cv2.LINE_AA)

                    cv2.namedWindow("Reidentificator", cv2.WINDOW_NORMAL)
                    cv2.imshow("Reidentificator", rgb)

                    if cv2.waitKey(1) == ord('q'):
                        print("Closed Reidentificator Image Viewer.")
                        exit(0)

        # KeyError exception rise up if no boxes and masks are found in inference input
        except KeyError:
            reidentified_class = None

        return reidentified_class

    ### ONLY USED FOR STATISTICS
    def calibrate_person_statistics(self, rgb):
        '''
        Function used to calibrate the reidentificator with the object image. This function should be called iteratively
        until it returns True (i.e., when the object is calibrated)
        :param rgb: the image in which there is the object
        :param inference_output: a dictionary containing the inferences obtained by an instance segmentation algorithm
        (e.g., Yolact++)
        :return: A boolean which confirm if the object has been correctly calibrated or not
        '''

        if self.iteration_number >= self.required_calibration_measures:
            if not self.calibrated:

                self.mean_pers = np.mean(np.array(self.feats), axis=0)
                self.std_pers = np.std(np.array(self.feats), axis=0)
                self.mahalanobis_deviation_const = np.sqrt(self.mean_pers.shape[0])
                self.calibrated = True
                print('\nCALIBRATION FINISHED\n')

            if self.iteration_number - self.required_calibration_measures < 100:
                percentage = self.iteration_number - self.required_calibration_measures / 100 * 100
                if percentage % 10 == 0:
                    print("THRESHOLD COMPUTATION ", int(percentage), "%")
                img_person = self.transform(torch.from_numpy(rgb).unsqueeze(0).cuda().float())
                self.iteration_number += 1
                # feat_pers = self.model_REID(img_person.cuda()).data.cpu()[0].numpy()
                feat_pers = self.model_REID(img_person).data.cpu()
                # pdb.set_trace()
                dist = np.linalg.norm((feat_pers - self.mean_pers) / (self.mahalanobis_deviation_const * self.std_pers),
                                      axis=1)

                self.feats_distances.append(dist)
            else:
                # plt.plot(np.arange(len(self.feats_distances)), np.array(self.feats_distances))
                # plt.plot(np.ones(len(self.feats_distances)) * np.mean(np.array(self.feats_distances)))
                # plt.plot(np.ones(len(self.feats_distances)) * (np.mean(np.array(self.feats_distances)) +
                #                                                2.2 * np.std(np.array(self.feats_distances))))
                # plt.legend(["Thresholds", "Mean", "2.2 std"])
                # plt.grid(True)
                # plt.show()

                self.feature_threshold = np.mean(np.array(self.feats_distances)) + \
                                         self.std_dev_confidence * np.std(np.array(self.feats_distances))
                print("\nTHRESHOLD: %.4f" % self.feature_threshold)
                self.calibration_finished = True

        else:  # CALIBRATION
            percentage = self.iteration_number / self.required_calibration_measures * 100
            if percentage % 10 == 0:
                print("CALIBRATING ", int(percentage), "%")
            img_person = self.transform(torch.from_numpy(rgb).unsqueeze(0).cuda().float())
            self.iteration_number += 1
            self.feats.append(self.model_REID(img_person.cuda()).data.cpu()[0].numpy())

        return self.calibration_finished

    def reidentify_statistics(self, rgb):
        '''
        Used to reidentify the calibrated object on the image (if present)
        :param rgb: the image in which there should be the object to reidentify
        :param inference_output: a dictionary containing the inferences obtained by an instance segmentation algorithm
        (e.g., Yolact++)
        :return: the mask of the targeted object if reidentified
        '''
        rgb = rgb.copy()
        if not self.calibrated:
            sys.exit("Error: Reidentificator not calibrated!")

        img_persons = self.transform(torch.from_numpy(rgb).unsqueeze(0).cuda().float())
        # PASS THE IMAGES INSIDE THE EXTERNAL NETWORK
        feat_pers = self.model_REID(img_persons).data.cpu()
        # COMPUTE FEATURES DISTANCES
        dist = np.linalg.norm((feat_pers - self.mean_pers) / (self.mahalanobis_deviation_const * self.std_pers),
                              axis=1)
        # print(dist)
        # RETURN REIDENTIFIED CLASS IFF THE DISTANCE BETWEEN FEATURE IS NO MORE THAN A THRESHOLD
        if dist > self.feature_threshold:
            reidentified_class = False
        else:
            reidentified_class = True


        return reidentified_class


