import cv2
import numpy as np
import matplotlib.pyplot as plt
from pyclothoids import Clothoid

def draw_axes(img, corners, ids, camera_matrix, dist_coeffs):
    marker_positions = []
    for corner, id in zip(corners, ids):
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corner, 0.05, camera_matrix, dist_coeffs)
        marker_x = tvec[0][0][0]
        marker_y = tvec[0][0][1]
        marker_positions.append((marker_x, marker_y))
        axis_points = np.float32([[0,0,0], [0.1,0,0], [0,0.1,0], [0,0,0.1]]).reshape(-1,3)
        imgpts, jac = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
        start = tuple(imgpts[0].ravel().astype(int))
        end_x = tuple(imgpts[1].ravel().astype(int))
        end_y = tuple(imgpts[2].ravel().astype(int))
        end_z = tuple(imgpts[3].ravel().astype(int))
        img = cv2.line(img, start, end_x, (255, 0, 0), 5)
        img = cv2.line(img, start, end_y, (0, 255, 0), 5)
        img = cv2.line(img, start, end_z, (0, 0, 255), 5)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f'X: {marker_x:.2f}m, Y: {marker_y:.2f}m', (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return img, marker_positions

def main():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    fx = 220.36842196079996
    fy = 253.76768534003318
    cx = 311.5594773936248
    cy = 177.2608792963969
    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([0.11594790711648671, 0.05201063162164969, -0.0035357590750148965, -0.028682986959198388, -0.009460180368416088], dtype=np.float32)
    cap = cv2.VideoCapture(0)
    marker_positions = []
    plt.figure(figsize=(10, 10))
    plt.ion()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            frame, positions = draw_axes(frame, corners, ids, camera_matrix, dist_coeffs)
            marker_positions.extend(positions)
            
            if marker_positions:
                marker_x, marker_y = marker_positions[-1]
                x_start = marker_x
                y_start = marker_y
                theta_start = 1.57
                x_end = 0.0
                y_end = 0.0
                theta_end = 1.57
                clothoid0 = Clothoid.G1Hermite(x_start, y_start, theta_start, x_end, y_end, theta_end)
                x_samples, y_samples = clothoid0.SampleXY(500)
                plt.clf()
                plt.plot(x_samples, y_samples, label='Clothoid curve')
                plt.scatter([x_start, x_end], [y_start, y_end], color='red')
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.title('Clothoid Curve from Start to End')
                plt.grid()
                plt.axis('equal')
                plt.legend()
                plt.pause(0.001)
        
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

