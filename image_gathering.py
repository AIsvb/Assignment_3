import cv2

def get_corner_frames(path, dest):
    vid = cv2.VideoCapture(path)
    out = cv2.VideoWriter(dest, cv2.VideoWriter_fourcc("M", "J", "P", "G"), 10, (644, 486))

    while vid.isOpened():
        ret, frame = vid.read()
        ret, corners = cv2.findChessboardCorners(frame, (8, 6), None)

        if ret:
            out.write(frame)
        cv2.imshow("img", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            vid.release()
            out.release()

def play_video(path):
    vid = cv2.VideoCapture(path)

    while vid.isOpened():
        ret, frame = vid.read()
        ret, corners = cv2.findChessboardCorners(frame, (8, 6), None)

        cv2.drawChessboardCorners(frame, (8, 6), corners, ret)
        cv2.imshow("img", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            vid.release()

def get_frames(path, dest):
    vid = cv2.VideoCapture(path)
    n = 1
    while vid.isOpened():
        ret, frame = vid.read()
        cv2.imshow("img", frame)

        if cv2.waitKey(1) & 0xFF == ord("s"):
            print(n)
            cv2.imwrite(dest + f"frame_{n}.png", frame)
            n += 1
        elif cv2.waitKey(1) & 0xFF == ord("q"):
            vid.release()

def show_images(files):
    for f in files:
        img = cv2.imread(f)
        ret, corners = cv2.findChessboardCorners(img, (8, 6), None)

        if not ret:
            print(f)
        cv2.drawChessboardCorners(img, (8, 6), corners, ret)
        cv2.imshow("img", img)
        cv2.waitKey(0)

path = "data/cam/intrinsics.avi"
dest = "data/cam4/corners.avi"

get_corner_frames(path, dest)
get_frames(dest, "data/cam4/calibration/")
# images = glob.glob("data/cam4/calibration/fra*")
# show_images(images)
play_video(dest)
cv2.destroyAllWindows()

