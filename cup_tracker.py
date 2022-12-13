from MultiMsgSync import TwoStageHostSeqSync
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import depthai as dai
import numpy as np
import json
import blobconverter
import keyboard
import time
from deep_sort_realtime.deepsort_tracker import DeepSort

NUM_CUPS = 6


identifiers = []
locations = [0]
track_id = 0
choice_flag = 0
frame_flag = 1
user_choice = 0
cycle_counter = 0
index = 0
cup_flag = []
lost_cup_buffer = []  # Potentially use a buffer for multiple lost cups but might screw things up
list_to_write = []
time_counter = 0
time_window = []
avg_time = 0
old_time = time.time()
order = []
max = -99999

print("DepthAI version", dai.__version__)
def frame_norm(frame, bbox):

    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


def decode_dets(detections, shape):
    bbs = []
    W, H = shape
    for detection in detections:
        x1, y1, x2, y2 = int(detection.xmin*W), int(detection.ymin*H), int(detection.xmax*W), int(detection.ymax*H)
        w, h = x2-x1, y2-y1
        bbs.append(([x1, y1, w, h], detection.confidence, detection.label, [x1/W, y1/H]))
    return bbs


def create_pipeline(stereo, yolo_config):
    pipeline = dai.Pipeline()

    print("Creating Color Camera...")
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(1080, 1080)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam.setPreviewNumFramesPool(60)

    cam_xout = pipeline.create(dai.node.XLinkOut)
    cam_xout.setStreamName("color")
    cam.preview.link(cam_xout.input)

    # ImageManip will resize the frame before sending it to the Object detection NN node
    obj_det_manip = pipeline.create(dai.node.ImageManip)
    obj_det_manip.initialConfig.setResize(640, 640)
    # obj_det_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.RGB888p)
    obj_det_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
    obj_det_manip.setMaxOutputFrameSize(640 * 640 * 3) # assume 3 channels UINT8 images
    cam.preview.link(obj_det_manip.inputImage)

    if stereo:
        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)

        monoRight = pipeline.create(dai.node.MonoCamera)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        # Spatial Detection network if OAK-D
        print("OAK-D detected, app will display spatial coordiantes")
        obj_det = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
        obj_det.setBoundingBoxScaleFactor(0.3)
        obj_det.setDepthLowerThreshold(100)
        obj_det.setDepthUpperThreshold(5000)
        stereo.depth.link(obj_det.inputDepth)
    else: # Detection network if OAK-1
        print("OAK-1 detected, app won't display spatial coordiantes")
        obj_det = pipeline.create(dai.node.YoloDetectionNetwork)

    metadata = yolo_config.get("NN_specific_metadata", {})

    obj_det.setBlobPath(blobconverter.from_zoo(name="yolov6n_coco_640x640", zoo_type="depthai", shaves=6, use_cache=False))
    confidence = metadata.get("confidence_threshold", config.get("confidence_threshold", 0.15))
    obj_det.setConfidenceThreshold(confidence)
    obj_det.setNumClasses(metadata["classes"])
    obj_det.setCoordinateSize(metadata["coordinates"])
    obj_det.setAnchors(metadata["anchors"])
    obj_det.setAnchorMasks(metadata["anchor_masks"])
    obj_det.setIouThreshold(metadata["iou_threshold"])
    obj_det.input.setBlocking(False)
    obj_det.input.setQueueSize(1)

    obj_det_manip.out.link(obj_det.input)

    # Send object detections to the host (for bounding boxes)
    obj_det_xout = pipeline.create(dai.node.XLinkOut)
    obj_det_xout.setStreamName("detection")
    obj_det.out.link(obj_det_xout.input)

    # Script node will take the output from the obj detection NN as an input and set ImageManipConfig
    # to the 'embedding_manip' to crop the initial frame
    image_manip_script = pipeline.create(dai.node.Script)
    obj_det.out.link(image_manip_script.inputs['detections'])
    # Remove in 2.18 and use `imgFrame.getSequenceNum()` in Script node
    obj_det.passthrough.link(image_manip_script.inputs['passthrough'])
    cam.preview.link(image_manip_script.inputs['preview'])

    image_manip_script.setScript("""
    import time
    msgs = dict()
    def add_msg(msg, name, seq = None):
        global msgs
        if seq is None:
            seq = msg.getSequenceNum()
        seq = str(seq)
        # node.warn(f"New msg {name}, seq {seq}")
        # Each seq number has it's own dict of msgs
        if seq not in msgs:
            msgs[seq] = dict()
        msgs[seq][name] = msg
        # To avoid freezing (not necessary for this ObjDet model)
        if 30 < len(msgs):
            node.warn(f"Removing first element! len {len(msgs)}")
            msgs.popitem() # Remove first element
    def get_msgs():
        global msgs
        seq_remove = [] # Arr of sequence numbers to get deleted
        for seq, syncMsgs in msgs.items():
            seq_remove.append(seq) # Will get removed from dict if we find synced msgs pair
            # node.warn(f"Checking sync {seq}")
            # Check if we have both detections and color frame with this sequence number
            if len(syncMsgs) == 2: # 1 frame, 1 detection
                for rm in seq_remove:
                    del msgs[rm]
                # node.warn(f"synced {seq}. Removed older sync values. len {len(msgs)}")
                return syncMsgs # Returned synced msgs
        return None
    def correct_bb(bb):
        if bb.xmin < 0: bb.xmin = 0.001
        if bb.ymin < 0: bb.ymin = 0.001
        if bb.xmax > 1: bb.xmax = 0.999
        if bb.ymax > 1: bb.ymax = 0.999
        return bb
    while True:
        time.sleep(0.001) # Avoid lazy looping
        preview = node.io['preview'].tryGet()
        if preview is not None:
            add_msg(preview, 'preview')
        dets = node.io['detections'].tryGet()
        if dets is not None:
            # TODO: in 2.18.0.0 use dets.getSequenceNum()
            passthrough = node.io['passthrough'].get()
            seq = passthrough.getSequenceNum()
            add_msg(dets, 'dets', seq)
        sync_msgs = get_msgs()
        if sync_msgs is not None:
            img = sync_msgs['preview']
            dets = sync_msgs['dets']
            for i, det in enumerate(dets.detections):
                cfg = ImageManipConfig()
                correct_bb(det)
                cfg.setCropRect(det.xmin, det.ymin, det.xmax, det.ymax)
                # node.warn(f"Sending {i + 1}. det. Seq {seq}. Det {det.xmin}, {det.ymin}, {det.xmax}, {det.ymax}")
                cfg.setResize(224, 224)
                cfg.setKeepAspectRatio(False)
                node.io['manip_cfg'].send(cfg)
                node.io['manip_img'].send(img)
    """)

    embedding_manip = pipeline.create(dai.node.ImageManip)
    embedding_manip.initialConfig.setResize(224, 224)
    embedding_manip.setMaxOutputFrameSize(224 * 224 * 3)
    embedding_manip.inputConfig.setWaitForMessage(True)
    embedding_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
    image_manip_script.outputs['manip_cfg'].link(embedding_manip.inputConfig)
    image_manip_script.outputs['manip_img'].link(embedding_manip.inputImage)

    # Second stage Embedding NN
    print("Creating recognition Neural Network...")
    embedding_nn = pipeline.create(dai.node.NeuralNetwork)
    embedding_nn.setBlobPath(blobconverter.from_zoo(name="mobilenetv2_imagenet_embedder_224x224", zoo_type="depthai", shaves=6))
    embedding_manip.out.link(embedding_nn.input)

    recognition_xout = pipeline.create(dai.node.XLinkOut)
    recognition_xout.setStreamName("embedding")
    embedding_nn.out.link(recognition_xout.input)

    return pipeline


def tracker_iter(detections, embeddings, tracker, frame):
    height, width = frame.shape[:2]
    # Decode detections into bounding boxes
    object_bbs = decode_dets(detections, (width, height))
    # Calculate embeddings for each crop
    object_embeds = np.array([emb.getFirstLayerFp16() for emb in embeddings])
    # Track iteration
    object_tracks = tracker.update_tracks(object_bbs, embeds=object_embeds)
    return object_tracks

def location_sort(locations, user_choice):
    global list_to_write
    list_to_write = []
    for location in locations:
        if locations.index(location) != user_choice:
            list_to_write.append(location)
    list_to_write.append(locations(user_choice))


with dai.Device() as device:
    stereo = 1 < len(device.getConnectedCameras())

    with open('./yolov6n.json', 'r') as file:
        raw_config = json.load(file)
        config = raw_config.get("nn_config", {})
        mappings = raw_config.get("mappings", {})
        labels = mappings.get("labels", {})

    device.startPipeline(create_pipeline(stereo, config))

    tracker = DeepSort(max_age=1000, nn_budget=None, embedder=None, nms_max_overlap=0.2, max_cosine_distance=0.05)

    sync = TwoStageHostSeqSync()
    queues = {}
    # Create output queues
    for name in ["color", "detection", "embedding"]:
        queues[name] = device.getOutputQueue(name)



    while True:
        for name, q in queues.items():
            # Add all msgs (color frames, object detections and recognitions) to the Sync class.
            if q.has():
                sync.add_msg(q.get(), name)

        msgs = sync.get_msgs()
        if msgs is not None:
            frame = msgs["color"].getCvFrame()
            detections = msgs["detection"].detections
            embeddings = msgs["embedding"]

            # Update the tracker
            object_tracks = tracker_iter(detections, embeddings, tracker, frame)



            # For each tracking object
            for track in object_tracks:

                if not track.is_confirmed() or track.time_since_update > 1 or track.detection_id >= len(detections) or track.detection_id < 0:
                    if track.track_id in identifiers and track.time_since_update > 5:
                        cup_flag[identifiers.index(track.track_id)] = 0
                        print(f'{identifiers.index(track.track_id)+1}' + " out of frame")
                    continue
                time_window.append(time.time())
                if len(time_window) >= 25:
                    time_window.pop(0)
                    avg_time = round(1 / ((time_window[0] + time_window[-1]) / 50 - old_time / 25), 2)
                    old_time = time_window[0]
                    # print("Detection Rate: " + f'{avg_time}' + " Hz")
                # dtime = time.time() - old_time
                # if dtime != 0:
                #     print(1/dtime)
                # old_time = time.time()
                detection = detections[track.detection_id]
                # print(cup_flag)
                # print(identifiers)
                # print(index)
                # print(locations)
                if labels[detection.label] == 'cup':
                    if track.track_id not in identifiers and len(identifiers) < NUM_CUPS:
                        identifiers.append(track.track_id)
                        loc = detection.spatialCoordinates
                        # locations.append([round(-1*loc.z/1000, 4), round(loc.x/1000, 4), round(loc.y/1000, 4)])
                        locations.append([int(-1 * loc.z), int(loc.x), int(loc.y)])

                        cup_flag.append(1)

                    elif track.track_id not in identifiers and len(identifiers) == NUM_CUPS and 0 in cup_flag:
                        index = cup_flag.index(0)
                        # if cup_flag.count(0) == 1:
                        identifiers[index] = track.track_id
                        cup_flag[index] = 1

                    elif track.track_id in identifiers:
                        index = identifiers.index(track.track_id)
                        loc = detection.spatialCoordinates
                        cup_flag[index] = 1
                        # locations[index+1] = [round(-1*loc.z/1000, 4), round(loc.x/1000, 4), round(loc.y/1000, 4)]
                        locations[index+1] = [int(-1 * loc.z), int(loc.x), int(loc.y)]

                    if track.track_id not in identifiers:
                        continue

                    track_id = identifiers.index(track.track_id) + 1

                    bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))

                    x, y = bbox[0] + 10, bbox[1] + 30
                # Displaying
                #     if frame_flag == 1:
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (10, 245, 10), 2)

                    if frame_flag == 1:
                        cv2.putText(frame, f'Cup {track_id}', (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 8)
                        cv2.putText(frame, f'Cup {track_id}', (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)
                    # cv2.putText(frame, f'Label: {labels[detection.label] if labels != {} else detection.label}, Confidence: {detection.confidence*100:.2f}%', (x, y + 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 8)
                    # cv2.putText(frame, f'Label: {labels[detection.label] if labels != {} else detection.label}, Confidence: {detection.confidence*100:.2f}%', (x, y + 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 255, 255), 2)
                    if stereo:
                        # You could also get detection.spatialCoordinates.x and detection.spatialCoordinates.y coordinates
                        coords = "({:.2f}, ".format(-1*detection.spatialCoordinates.z/1000) + "{:.2f}, ".format(detection.spatialCoordinates.x/1000) + "{:.2f}) m".format(detection.spatialCoordinates.y/1000)
                        cv2.putText(frame, coords, (x, y + 60), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 8)
                        cv2.putText(frame, coords, (x, y + 60), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 255, 255), 2)

                        # print(len(identifiers))
                        # print(user_choice)
                        # print(locations[1])
                        # print(f'Cup {track_id}: ' + "X: {:.3f} m".format(detection.spatialCoordinates.x/1000) + "    Y: {:.3f} m".format(detection.spatialCoordinates.y/1000) + "    Z: {:.3f} m".format(detection.spatialCoordinates.z/1000) + '\n\n')

            if user_choice == 0:
                cv2.imshow("Camera", frame)

            if len(identifiers) == NUM_CUPS and cycle_counter < 10 and 0 not in cup_flag:
                cycle_counter = cycle_counter + 1
            elif choice_flag == 0 and 0 in cup_flag:
                cycle_counter = 0
            if len(identifiers) == NUM_CUPS and choice_flag == 0 and cycle_counter == 10:
                user_choice = user_choice = input("The ball is in cup: ")
                choice_flag = 1

            cv2.imshow("Camera", frame)
            for identifier in identifiers:
                ind = identifiers.index(identifier)
                print(locations[ind][0])
            # if keyboard.is_pressed('Space'):
            #     # cv2.imshow("Camera", frame)
            #     locations[0] = 1
            #     # list_to_write = location_sort(locations, user_choice)
            #     print('Archiving most recent locations....')
            #     file = open("cup_locations.txt", "w")
            #     file.write(str(locations[0]))
            #     file.close()

            #     for i in range(1, NUM_CUPS + 1):
            #         if int(i) != int(user_choice):
            #             # print(i)
            #             # print(user_choice)
            #             # print(i==user_choice)
            #             file = open("cup_locations.txt", "a")
            #             # print(" | " + str(locations[i]))
            #             file.write(" | " + str(locations[i]))
            #             file.close()
            #             continue
            #         # file.write(str(user_choice + "\n"))
            #     file = open("cup_locations.txt", "a")
            #     file.write(" | " + str(locations[int(user_choice)]))
            #     # print(" | " + str(locations[int(user_choice)]))
            #     file.close()
            #     break
            # if keyboard.is_pressed('x') or keyboard.is_pressed('X'):
            #     break
        if cv2.waitKey(1) == ord('q'):
            break
