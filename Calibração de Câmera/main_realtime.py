import cv2 
import numpy as np
import os

def cls():
    os.system('cls' if os.name == 'nt' else 'clear')

# Captação de video
cap = cv2.VideoCapture(1)
cap.set(3,1280)
cap.set(4,720)

if not cap.isOpened():
    print(" Cannot open camera")
    exit() 
  
# Define as dimensões do tabuleiro de xadrex e os criterios de rescisão.
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) 
CHECKERBOARD = (7, 7)  
  
# Define os pontos do mundo, cordernadas, para os objetos em três dimensões.
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32) 
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) 

# Cria um vector para armazenar os vectors dos objetos em três dimensões e dos objetos em duas dimensões das images do tabuleiro de xadrex.  
objpoints = []
imgpoints = [] 

# Receiving a individual image, frame, from camera  
while True:

    if cv2.waitKey(1) == ord('q'):
        break

    ret, image = cap.read()

    grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

   # Caso o número de cantos for detectado, busca as coordenadas de pixel e as exibe nas imagens do tabuleiro de xadrez.
    ret, corners = cv2.findChessboardCorners(grayColor, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE) 
  
    if ret == True: 

        objpoints.append(objp) 
        
        # Agrupando as coordenadas de pixel em pontos 2D.
        corners2 = cv2.cornerSubPix(grayColor, corners, (11, 11), (-1, -1), criteria) 
        imgpoints.append(corners2) 

        if(len(imgpoints) == 150):
            break;

        image = cv2.drawChessboardCorners(image, CHECKERBOARD,corners2, ret) 

        # Desenha e exibe os cantos 2D.
        cv2.imshow('Camera Calibration', image)
        cv2.waitKey(150)

    else:
        cls()
        print("Failed to find a chessboard")

cv2.destroyAllWindows()
cls()
print(" Calculating...") 
  
h, w = image.shape[:2] 
  
# Realizando a calibração da câmera passando o valor dos pontos 3D conhecidos e as coordenadas de pixel correspondentes dos cantos detectados 2D.
ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(objpoints, imgpoints, grayColor.shape[::-1], None, None) 

# Calcula o error de reprojeção. 
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], r_vecs[i], t_vecs[i], matrix, distortion)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

mean_error = mean_error/len(objpoints)

print(" Camera matrix:") 
print(matrix) 
  
print("\n Distortion coefficient:") 
print(distortion)

print("\n Re-projection Error:") 
print(mean_error)

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(matrix, distortion, (w,h), 1, (w,h))

while True:

    if cv2.waitKey(1) == ord('q'):
        break
    
    ret, frame = cap.read()

    dst = cv2.undistort(frame, matrix, distortion, None, newcameramtx)

    if not ret:
        break

    x, y, w, h = roi

    dst = dst[y:y+h, x:x+w]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Camera Calibration', dst)

cap.release()
cv2.destroyAllWindows()
exit()