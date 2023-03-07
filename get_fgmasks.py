# Computer Vision: Assignment 2
# Creators: Gino Kuiper and Sander van Bennekom
# Date: 04-03-2023

if __name__ == "__main__":
    from BackgroundSubtractor import BackgroundSubtractor as BS

    bs1 = BS('data/cam1/background.avi', 'data/cam1/video.avi')
    # bs2 = BS('data/cam2/background.avi', 'data/cam2/video.avi')
    # bs3 = BS('data/cam3/background.avi', 'data/cam3/video.avi')
    # bs4 = BS('data/cam4/background.avi', 'data/cam4/video.avi')

    bs1.subtract_background()
    # bs2.subtract_background()
    # bs3.subtract_background()
    # bs4.subtract_background()
