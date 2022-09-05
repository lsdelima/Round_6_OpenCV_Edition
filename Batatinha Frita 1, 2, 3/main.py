from playsound import playsound
import mediapipe as mp
import cvzone as cvz
import numpy as np
import math
import time
import cv2

#Função polinomial que calcula a taxa de error a ser ignorada.
y = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
x = [45, 55, 65, 35, 85, 135, 145]
coff = np.polyfit(x, y, 2)

#Definir o tamanho da imagem
width, height = 1280, 720

#inicialização do tempo e variáveis de uso no jogo.
timestart = time.time()
timegreen = time.time()
totaltime = 40

doll_g_img = cv2.imread("green.png")
doll_r_img = cv2.imread("red.png")

blood_img = cv2.imread("blood.png")

testpos = dis = real = 0

play_song = True
red = timesleep = dead = False

#função que detecta a diferença de movimentação dado um frame anterior.
def motion_detection(testpos, pos):
    
    for id in range(0,32):
        if( ((pos[id].x > testpos[id].x + real) or (pos[id].x < testpos[id].x - real)) or ((pos[id].y > testpos[id].y + real) or (pos[id].y < testpos[id].y - real))):
            return True

        else:
            continue

    return False

#função que detectar a distância aproximada do corpo da câmera, dado a distância euclidiana de dois pontos, os ombros.
def distance_detection(landmarks):
    
    x1 = x2 = y1 = y2 = 0

    for id, lm in enumerate(landmarks):

        cx, cy = lm.x * width, lm.y * height

        if id == 11:
            x1, y1 = cx, cy

        elif id == 12:
            x2,y2 = cx, cy

    distance = int(math.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2 ))

    return distance

#função que retorna a posição do corpo na imagem.
def position(results):

    try:
        landmarks = results.pose_landmarks.landmark
        distance = distance_detection(landmarks)
        return landmarks, distance

    except:
        exit()

#Importação das função de body Tracking do mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#inicialização da camêra e da função de captura de imagem
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

pose = mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5)

#código do jogo
while True:

    #tocador de musica e opção de saida
    if play_song == True:
        playsound("doll.mp3", False)
        play_song = False

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    #Captação de imagem e extração da posição do corpo, jogador.
    ret, frame = cap.read()

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    results = pose.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    #jogo; tempo = 40 segundo, caso o jogaor chegue antes e não perca, o mesmo ganha.
    if time.time() - timestart < totaltime and dead == False and dis < 200:

        _ , dis = position(results)

        if(red == True):

            cvz.putTextRect(doll_r_img, f'00:{int(totaltime-(time.time()-timestart))}',[55, 655], colorR = (0,0,0), scale=5, offset=40)

            cv2.imshow("Batatinha Frita 1, 2, 3",doll_r_img)

            pos, _ = position(results)

            dead = motion_detection(testpos, pos)

            if time.time() - timegreen >= 5:
                timegreen = time.time()
                red = False
                play_song = True

        elif(red == False):

            cvz.putTextRect(doll_g_img, f'00:{int(totaltime-(time.time()-timestart))}',[55, 655], colorR = (0,0,0), scale=5, offset=40)

            cv2.imshow("Batatinha Frita 1, 2, 3",doll_g_img)

            if time.time() - timegreen >= 2.5:
                testpos, _ = position(results)
                A , B, C = coff
                real = A * dis ** 2 + B * dis + C
                timegreen = time.time()
                red = True

        # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        #                             mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
        #                             mp_drawing.DrawingSpec(color=(66, 245, 66), thickness=2, circle_radius=2))

        # # cvz.putTextRect(image, f'00:{int(totaltime-(time.time()-timestart))}',[1080, 75], colorR = (0,0,0), scale=3, offset=20)
        # cv2.imshow('Mediapipe Feed', image)

    elif dis >= 200:
        playsound("win.mp3", False)
        cvz.putTextRect(image, f'Ganhou', (480, 400), scale=5, colorR = (32,165,218), offset=30, thickness=7)
        cv2.imshow("Batatinha Frita 1, 2, 3", image)
        if cv2.waitKey(0) == ord('q'):
            break

    else:
        playsound("gum_shot.wav", False)
        exit = cv2.addWeighted(image, 0.2, blood_img, 0.8, 0)
        cvz.putTextRect(exit, f'Fim de Jogo', (400, 400), scale=5, colorR = (0,0,0), offset=30, thickness=7)
        cv2.imshow("Batatinha Frita 1, 2, 3", exit)
        if cv2.waitKey(0) == ord('q'):
            break

#encerra o programa.
cap.release()
cv2.destroyAllWindows()
exit()