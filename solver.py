from network import *
from imgutils import *
import cv2
import numpy as np
import sudoku
from sudoku import SudokuPuzzle, depth_first_solve
import sys

net = Net()
net.load_state_dict(torch.load('cnet.p'))
net.eval()
solved = False
unsolved = []
solvedboard = []
solvedstr = ""
corners = []
img = []
prevcorners = []
counter = 0
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

_, prevframe = cap.read()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not solved:
        unsolved = getBoardArray(frame, net)
        if isValidBoard(unsolved):
            [list(map(str,i)) for i in unsolved]
            puzzle = SudokuPuzzle(9, [list(map(str,i)) for i in unsolved], {"1", "2", "3", "4", "5", "6", "7", "8", "9"})
            solution = depth_first_solve(puzzle)
            if solution is not None:
                solvedboard = [list(map(int,i)) for i in solution.get_symbols()]
                solvedstr = getAnsString(solvedboard, unsolved)
                print(solvedstr)
                solved = True
                counter = 0
            else:
                solved = False
        
    
    elif len(solvedboard) > 7 and len(corners) > 0 and counter < 2 and solved:
        if len(prevcorners) > 0 and int(prevcorners[0][0]) == int(corners[0][0]):
            counter += 1
        ln, ht, _ = img.shape
        x, y = (10, int(ht / 9) - 25)
        image = np.zeros((2*ln, 2*ln, 3),np.uint8)
        for s in solvedstr:
            if s == "\n":
                x = 0
                y += int(ht / 9)
            else:
                cv2.putText(image, s, (x,y), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255,255), 3)
                x += int(ln / 9)
        src = np.float32([[0, 0], [0, ht], [ln, ht], [ln, 0]])
        try:
            dst = corners * (ht / ln * 0.9)
            M = cv2.getPerspectiveTransform(src, dst)
            image = cv2.warpPerspective(image, M, (frame.shape[1], frame.shape[0]))
            prevframe[image[:, :, 2] > 0] = [255, 0, 0]
        except cv2.error:
            pass
    else:
        solved = False
    
    cv2.imshow('Sudoku Solver', prevframe)
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    x = preprocess(frame)
    prevcorners = corners.copy()
    if x is not None:
        mask, corners, img = x
        counter = 0
    
    #if solved: break
    
    if cv2.waitKey(1) == ord('q'):
        break
        
    prevframe = frame
        
        
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
