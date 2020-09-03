import cv2
import argparse
import math
import numpy as np
import os
import glob
import xlsxwriter
# import pandas as pd

CORNER_FEATS = (
    0.322965313273202,
    0.19188334690998524,
    1.1514327482234812,
    0.998754685666376,
    )

def normalize(im):
    return cv2.normalize(im, np.zeros(im.shape), 1, 255, norm_type=cv2.NORM_MINMAX)

def get_approx_contour(contour, tol=.03):
    """Get rid of 'useless' points in the contour"""
    epsilon = tol * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)

def get_contours(image_gray):
    _,contours,_ = cv2.findContours(image_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return map(get_approx_contour, contours)

def get_corners(contours):
    return sorted(
        contours,
        key=lambda c: features_distance(CORNER_FEATS, get_features(c)))[:4]

def get_bounding_rect(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return np.int0(box)

def get_convex_hull(contour):
    return cv2.convexHull(contour)

def get_contour_area_by_hull_area(contour):
    return (cv2.contourArea(contour) /
            cv2.contourArea(get_convex_hull(contour)))

def get_contour_area_by_bounding_box_area(contour):
    return (cv2.contourArea(contour) /
            cv2.contourArea(get_bounding_rect(contour)))

def get_contour_perim_by_hull_perim(contour):
    return (cv2.arcLength(contour, True) /
            cv2.arcLength(get_convex_hull(contour), True))

def get_contour_perim_by_bounding_box_perim(contour):
    return (cv2.arcLength(contour, True) /
            cv2.arcLength(get_bounding_rect(contour), True))

def get_features(contour):
    try:
        return (
            get_contour_area_by_hull_area(contour),
            get_contour_area_by_bounding_box_area(contour),
            get_contour_perim_by_hull_perim(contour),
            get_contour_perim_by_bounding_box_perim(contour),
        )
    except ZeroDivisionError:
        return 4*[np.inf]

def features_distance(f1, f2):
    return np.linalg.norm(np.array(f1) - np.array(f2))

# Default mutable arguments should be harmless here

def draw_point(point, img, radius=6, color=(0, 0, 255)):
    cv2.circle(img, tuple(point), radius, color, -1)

def get_centroid(contour):
    m = cv2.moments(contour)
    x = int(m["m10"] / m["m00"])
    y = int(m["m01"] / m["m00"])
    return (x, y)

def order_points(points):
    """Order points counter-clockwise-ly."""
    origin = np.mean(points, axis=0)

    def positive_angle(p):
        x, y = p - origin
        ang = np.arctan2(y, x)
        return 2 * np.pi + ang if ang < 0 else ang

    return sorted(points, key=positive_angle)

def get_outmost_points(contours):
    all_points = np.concatenate(contours)
    return get_bounding_rect(all_points)

def perspective_transform(img, points):
    """Transform img so that points are the new corners"""

    source = np.array(
        points,
        dtype="float32")

    dest = np.array([
        [800, 600],
        [0, 600],
        [0, 0],
        [800, 0]],
        dtype="float32")

    img_dest = img.copy()
    transf = cv2.getPerspectiveTransform(source, dest)
    warped = cv2.warpPerspective(img, transf, (800, 600))
    return warped

def sheet_coord_to_transf_coord(x, y):
    return list(map(lambda n: int(np.round(n)), (
        800 * x/744.055,
        600 * (1 - y/1052.362)
    )))
    #TRANSF_SIZE * x/744.055,
        #TRANSF_SIZE * (1 - y/1052.362)


# STUDENT ERP

def get_erp_patch(transf, q_number):
    #Top left
    tl = sheet_coord_to_transf_coord(
        65 - 19.5 * (1 - q_number),
        675
    )
     #Bottom right
    br = sheet_coord_to_transf_coord(
        78 - 19.5 * (1 - q_number ),
        295
    )
    
    return transf[tl[1]:br[1], tl[0]:br[0]]

def get_erp_patches(transf):
    for i in range(1, 11):
        yield get_erp_patch(transf, i)

def get_erp_alternative_patches(erp_patch):
    for i in range(10):
        x0, _ = sheet_coord_to_transf_coord(20 * i, 0)
        x1, _ = sheet_coord_to_transf_coord(20 + 20 * i, 0)
        yield erp_patch[x0:x1,:]
        
def draw_marked_erp_alternative(erp_patch, index):
    cx, cy = sheet_coord_to_transf_coord(
        15/2,
        20 * (2 * index + .5))
    draw_point((cx,600-cy), erp_patch, radius=7, color=(255, 0, 0))

def get_marked_erp_alternative(erp_alternative_patches):
    means = list(map(np.mean, erp_alternative_patches))
    sorted_means = sorted(means)

    # Simple heuristic
    if sorted_means[0]/sorted_means[1] > 1 :
        return None

    return np.argmin(means)

def get_letter(erp_index):
    return ["1", "2", "3", "4", "5","6","7","8","9","0"][erp_index] if erp_index is not None else "n/a"

#STUDENT TEST ID

def get_testid_patch(transf, q_number):
    #Top left
    tl = sheet_coord_to_transf_coord(
        320 - 19.5 * (1 - q_number),
        675
    )
     #Bottom right
    br = sheet_coord_to_transf_coord(
        335 - 19.5 * (1 - q_number ),
        295
    )
    return transf[tl[1]:br[1], tl[0]:br[0]]

def get_testid_patches(transf):
    for i in range(1, 6):
        yield get_testid_patch(transf, i)

def get_alternative_testid_patches(testid_patch):
    for i in range(10):
        x0, _ = sheet_coord_to_transf_coord(20 * i, 0)
        x1, _ = sheet_coord_to_transf_coord(16 + 20 * i, 0)
        yield testid_patch[x0:x1,:]
        
def draw_marked_testid_alternative(testid_patch, index):
    cx, cy = sheet_coord_to_transf_coord(
        15/2,
        20 * (2 * index + .5))
    draw_point((cx,600-cy), testid_patch, radius=7, color=(255, 0, 0))

def get_marked_testid_alternative(testid_patches):
    means = list(map(np.mean, testid_patches))
    sorted_means = sorted(means)

    # Simple heuristic
    if sorted_means[0]/sorted_means[1] > 1:
        return None

    return np.argmin(means)

def get_letter(testid_index):
    return ["1", "2", "3", "4", "5","6","7","8","9","0"][testid_index] if testid_index is not None else "n/a"

#STUDENT 1st 5 MARKED ANSWERS

def get_question_patch(transf, q_number):
    # Top left
    tl = sheet_coord_to_transf_coord(
        490,
        678 - 78 * (q_number - 1)
    )
    # Bottom right
    br = sheet_coord_to_transf_coord(
        570,
        650- 78 * (q_number - 1)
    )
    
    return transf[tl[1]:br[1], tl[0]:br[0]]

def get_question_patches(transf):
    for i in range(1, 6):
        yield get_question_patch(transf, i)

def get_alternative_patches(question_patch):
    for i in range(4):
        x0, _ = sheet_coord_to_transf_coord(20 * i, 0)
        x1, _ = sheet_coord_to_transf_coord(10 + 20 * i, 0)
        yield question_patch[:, x0:x1]

def draw_marked_alternative(question_patch, index):
    cx, cy = sheet_coord_to_transf_coord(
        10 * (2 * index + .5),
        30/2)
    draw_point((cx, 600-cy), question_patch, radius=7, color=(255, 0, 0))

def get_marked_alternative(alternative_patches):
    means = list(map(np.mean, alternative_patches))
    sorted_means = sorted(means)

    # Simple heuristic
    if sorted_means[0]/sorted_means[1] > .9:
        return None

    return np.argmin(means)

def get_let(alt_index):
    return ["A", "B", "C", "D"][alt_index] if alt_index is not None else "N/A"

#STUDENT 2ND 5 MARKED ANSWERS

def get_secquestion_patch(transf, q_number):
    # Top left
    tl = sheet_coord_to_transf_coord(
        610,
        678 - 78 * (q_number - 1)
    )
 
    # Bottom right
    br = sheet_coord_to_transf_coord(
        690,
        650- 78 * (q_number - 1)
    )
    
    return transf[tl[1]:br[1], tl[0]:br[0]]

def get_secquestion_patches(transf):
    for i in range(1, 6):
        yield get_secquestion_patch(transf, i)

def get_secalternative_patches(secquestion_patch):
    for i in range(4):
        x0, _ = sheet_coord_to_transf_coord(20 * i, 0)
        x1, _ = sheet_coord_to_transf_coord(10 + 20 * i, 0)
        yield secquestion_patch[:, x0:x1]

def draw_marked_secalternative(secquestion_patch, index):
    cx, cy = sheet_coord_to_transf_coord(
        #10.8 * (2 * index + .5)
        10.8 * (2 * index + .5),
        30/2)
    draw_point((cx, 600-cy), secquestion_patch, radius=7, color=(255, 0, 0))

def get_marked_secalternative(secalternative_patches):
    means = list(map(np.mean, secalternative_patches))
    sorted_means = sorted(means)

    # Simple heuristic
    if sorted_means[0]/sorted_means[1] > .9:
        return None

    return np.argmin(means)

def get_lett(alt_indexx):
    return ["A", "B", "C", "D"][alt_indexx] if alt_indexx is not None else "N/A"

#STUDENT 3rd 2 MARKED ANSWERS

def get_trdquestion_patch(transf, q_number):
    # Top left
    tl = sheet_coord_to_transf_coord(
        65,
        195 - 78 * (q_number - 1)
    )
    # Bottom right
    br = sheet_coord_to_transf_coord(
        140,
        170- 78 * (q_number - 1)
    )
    
    return transf[tl[1]:br[1], tl[0]:br[0]]

def get_trdquestion_patches(transf):
    for i in range(1, 3):
        yield get_trdquestion_patch(transf, i)

def get_trdalternative_patches(trdquestion_patch):
    for i in range(4):
        x0, _ = sheet_coord_to_transf_coord(20 * i, 0)
        x1, _ = sheet_coord_to_transf_coord(10 + 20 * i, 0)
        yield trdquestion_patch[:, x0:x1]

def draw_marked_trdalternative(trdquestion_patch, index):
    cx, cy = sheet_coord_to_transf_coord(
        10 * (2 * index + .5),
        30/2)
    draw_point((cx, 600-cy), trdquestion_patch, radius=7, color=(255, 0, 0))

def get_marked_trdalternative(trdalternative_patches):
    means = list(map(np.mean, trdalternative_patches))
    sorted_means = sorted(means)

    # Simple heuristic
    if sorted_means[0]/sorted_means[1] > .9:
        return None

    return np.argmin(means)

def get_lettt(alt_indexxx):
    return ["A", "B", "C", "D"][alt_indexxx] if alt_indexxx is not None else "N/A"

#STUDENT 4th 2 MARKED ANSWERS

def get_fouquestion_patch(transf, q_number):
    # Top left
    tl = sheet_coord_to_transf_coord(
        200,
        195 - 78 * (q_number - 1)
    )
    # Bottom right
    br = sheet_coord_to_transf_coord(
        280,
        170- 78 * (q_number - 1)
    )
    
    return transf[tl[1]:br[1], tl[0]:br[0]]

def get_fouquestion_patches(transf):
    for i in range(1, 3):
        yield get_fouquestion_patch(transf, i)

def get_foualternative_patches(fouquestion_patch):
    for i in range(4):
        x0, _ = sheet_coord_to_transf_coord(20 * i, 0)
        x1, _ = sheet_coord_to_transf_coord(10 + 20 * i, 0)
        yield fouquestion_patch[:, x0:x1]

def draw_marked_foualternative(fouquestion_patch, index):
    cx, cy = sheet_coord_to_transf_coord(
        10 * (2 * index + .5),
        30/2)
    draw_point((cx, 600-cy), fouquestion_patch, radius=7, color=(255, 0, 0))

def get_marked_foualternative(foualternative_patches):
    means = list(map(np.mean, foualternative_patches))
    sorted_means = sorted(means)

    # Simple heuristic
    if sorted_means[0]/sorted_means[1] > .9:
        return None

    return np.argmin(means)

def get_letx(alt_idx):
    return ["A", "B", "C", "D"][alt_idx] if alt_idx is not None else "N/A"

#STUDENT 5th 2 MARKED ANSWERS

def get_fivquestion_patch(transf, q_number):
    # Top left
    tl = sheet_coord_to_transf_coord(
        345,
        195 - 78 * (q_number - 1)
    )
    # Bottom right
    br = sheet_coord_to_transf_coord(
        420,
        170- 78 * (q_number - 1)
    )
    
    return transf[tl[1]:br[1], tl[0]:br[0]]

def get_fivquestion_patches(transf):
    for i in range(1, 3):
        yield get_fivquestion_patch(transf, i)

def get_fivalternative_patches(fivquestion_patch):
    for i in range(4):
        x0, _ = sheet_coord_to_transf_coord(20 * i, 0)
        x1, _ = sheet_coord_to_transf_coord(10 + 20 * i, 0)
        yield fivquestion_patch[:, x0:x1]

def draw_marked_fivalternative(fivquestion_patch, index):
    cx, cy = sheet_coord_to_transf_coord(
        10 * (2 * index + .5),
        30/2)
    draw_point((cx, 600-cy), fivquestion_patch, radius=7, color=(255, 0, 0))

def get_marked_fivalternative(fivalternative_patches):
    means = list(map(np.mean, fivalternative_patches))
    sorted_means = sorted(means)

    # Simple heuristic
    if sorted_means[0]/sorted_means[1] > .9:
        return None

    return np.argmin(means)

def get_lettxx(alt_idxx):
    return ["A", "B", "C", "D"][alt_idxx] if alt_idxx is not None else "N/A"


#STUDENT 6th 2 MARKED ANSWERS

def get_sixquestion_patch(transf, q_number):
    # Top left
    tl = sheet_coord_to_transf_coord(
        485,
        195 - 78 * (q_number - 1)
    )
    # Bottom right
    br = sheet_coord_to_transf_coord(
        560,
        170- 78 * (q_number - 1)
    )
    
    return transf[tl[1]:br[1], tl[0]:br[0]]

def get_sixquestion_patches(transf):
    for i in range(1, 3):
        yield get_sixquestion_patch(transf, i)

def get_sixalternative_patches(sixquestion_patch):
    for i in range(4):
        x0, _ = sheet_coord_to_transf_coord(20 * i, 0)
        x1, _ = sheet_coord_to_transf_coord(10 + 20 * i, 0)
        yield sixquestion_patch[:, x0:x1]

def draw_marked_sixalternative(sixquestion_patch, index):
    cx, cy = sheet_coord_to_transf_coord(
        10 * (2 * index + .5),
        30/2)
    draw_point((cx, 600-cy), sixquestion_patch, radius=7, color=(255, 0, 0))

def get_marked_sixalternative(sixalternative_patches):
    means = list(map(np.mean, sixalternative_patches))
    sorted_means = sorted(means)

    # Simple heuristic
    if sorted_means[0]/sorted_means[1] > .9:
        return None

    return np.argmin(means)

def get_letsix(alt_idxsix):
    return ["A", "B", "C", "D"][alt_idxsix] if alt_idxsix is not None else "N/A"

#STUDENT 7th 2 MARKED ANSWERS

def get_sevquestion_patch(transf, q_number):
    # Top left
    tl = sheet_coord_to_transf_coord(
        610,
        195 - 78 * (q_number - 1)
    )
    # Bottom right
    br = sheet_coord_to_transf_coord(
        685,
        170- 78 * (q_number - 1)
    )
    
    return transf[tl[1]:br[1], tl[0]:br[0]]

def get_sevquestion_patches(transf):
    for i in range(1, 3):
        yield get_sevquestion_patch(transf, i)

def get_sevalternative_patches(sevquestion_patch):
    for i in range(4):
        x0, _ = sheet_coord_to_transf_coord(20 * i, 0)
        x1, _ = sheet_coord_to_transf_coord(10 + 20 * i, 0)
        yield sevquestion_patch[:, x0:x1]

def draw_marked_sevalternative(sevquestion_patch, index):
    cx, cy = sheet_coord_to_transf_coord(
        10 * (2 * index + .5),
        30/2)
    draw_point((cx, 600-cy), sevquestion_patch, radius=7, color=(255, 0, 0))

def get_marked_sevalternative(sevalternative_patches):
    means = list(map(np.mean, sevalternative_patches))
    sorted_means = sorted(means)

    # Simple heuristic
    if sorted_means[0]/sorted_means[1] > .8:
        return None

    return np.argmin(means)

def get_letsev(alt_idxsev):
    return ["A", "B", "C", "D"][alt_idxsev] if alt_idxsev is not None else "N/A"

def get_answers(img):  
    
    im_orig = cv2.imread(img)
    #im_orig= cv2.imread(image_path)
    blurred = cv2.GaussianBlur(im_orig, (11,11), 10)

    im = normalize(cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY))

    ret,im = cv2.threshold(im, 127 , 255, cv2.THRESH_BINARY)#127

    contours = get_contours(im)
    corners = get_corners(contours)

    cv2.drawContours(im_orig, corners, -1, (255, 0, 0), 3)

    outmost = order_points(get_outmost_points(corners))

    transf = perspective_transform(im_orig, outmost)
    erpp = []
    erp =[]
    e_ind=[]
    for i, q_patch in enumerate(get_erp_patches(transf)):
        erp_index = get_marked_erp_alternative(get_erp_alternative_patches(q_patch))
        if erp_index is not None:
            draw_marked_erp_alternative(q_patch, erp_index)
            erpp.append(get_letter(erp_index))
            erp = ''.join(erpp)     

    testidd = []
    testid=[]
    for i, q_patch in enumerate(get_testid_patches(transf)):
        testid_index = get_marked_testid_alternative(get_alternative_testid_patches(q_patch))

        if testid_index is not None:
            draw_marked_testid_alternative(q_patch, testid_index)

        testidd.append(get_letter(testid_index))
        testid = ''.join(testidd)     
    
    answers = []
    for i, q_patch in enumerate(get_question_patches(transf)):
        alt_index = get_marked_alternative(get_alternative_patches(q_patch))

        if alt_index is not None:
            draw_marked_alternative(q_patch, alt_index)

        answers.append(get_let(alt_index))

    secanswers = []
    for i, q_patch in enumerate(get_secquestion_patches(transf)):
        alt_indexx = get_marked_secalternative(get_secalternative_patches(q_patch))

        if alt_indexx is not None:
            draw_marked_secalternative(q_patch, alt_indexx)

        secanswers.append(get_lett(alt_indexx))
    
    trdanswers = []
    for i, q_patch in enumerate(get_trdquestion_patches(transf)):
        alt_indexxx = get_marked_trdalternative(get_trdalternative_patches(q_patch))

        if alt_indexxx is not None:
            draw_marked_trdalternative(q_patch, alt_indexxx)

        trdanswers.append(get_lettt(alt_indexxx))

        fouanswers = []
    for i, q_patch in enumerate(get_fouquestion_patches(transf)):
        alt_idx = get_marked_foualternative(get_foualternative_patches(q_patch))

        if alt_idx is not None:
            draw_marked_foualternative(q_patch, alt_idx)

        fouanswers.append(get_letx(alt_idx))

        fivanswers = []
    for i, q_patch in enumerate(get_fivquestion_patches(transf)):
        alt_idxx = get_marked_fivalternative(get_fivalternative_patches(q_patch))

        if alt_idxx is not None:
            draw_marked_fivalternative(q_patch, alt_idxx)

        fivanswers.append(get_lettxx(alt_idxx))

        sixanswers = []
    for i, q_patch in enumerate(get_sixquestion_patches(transf)):
        alt_idxsix = get_marked_sixalternative(get_sixalternative_patches(q_patch))

        if alt_idxsix is not None:
            draw_marked_sixalternative(q_patch, alt_idxsix)

        sixanswers.append(get_letsix(alt_idxsix))

        sevanswers = []
    for i, q_patch in enumerate(get_sevquestion_patches(transf)):
        alt_idxsev = get_marked_sevalternative(get_sevalternative_patches(q_patch))

        if alt_idxsev is not None:
            draw_marked_sevalternative(q_patch, alt_idxsev)

        sevanswers.append(get_letsev(alt_idxsev))
    return erp,testid,answers,secanswers,trdanswers,fouanswers,fivanswers,sixanswers,sevanswers,transf
    #print('X_data shape:', np.array(X_data))

def main():
    workbook = xlsxwriter.Workbook('bomb.xlsx') 
    worksheet = workbook.add_worksheet()
    row=0
    col=0 
    header_data = ['ERP', 'TEST_ID','Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10'
                    ,'Q11','Q12','Q13','Q14','Q15','Q16','Q17','Q18','Q19','Q20']
    header_format = workbook.add_format({'bold': True,'bottom': 2,'bg_color': '#F9DA04'})
    for col_num, data in enumerate(header_data):
        worksheet.write(0, col_num, data, header_format)
    ans =[]
    anss=[]
    path = r"input/*.jpg"
    for image in glob.glob(path):
        print(image)
    # img = os.path.join(path_of_images,image)
        erp,testid,answers,secanswers,trdanswers,fouanswers,fivanswers,sixanswers,sevanswers,im = get_answers(image)   
        print("ERP:{}".format(erp))
        worksheet.write(row+1,col, erp)
        print("Test_ID:{}".format(testid))
        worksheet.write(row+1,col+1, testid)
        print("ANSWERS:")
       
        for i, answer in enumerate(answers):
            print("Q{}: {}".format(i + 1, answer))
            worksheet.write(row+1,col+2,answers[0])
            worksheet.write(row+1,col+3,answers[1])
            worksheet.write(row+1,col+4,answers[2])
            worksheet.write(row+1,col+5,answers[3])
            worksheet.write(row+1,col+6,answers[4])
        for i, answerr in enumerate(secanswers):
            print("Q{}: {}".format(i + 6, answerr))
            worksheet.write(row+1,col+7,secanswers[0])
            worksheet.write(row+1,col+8,secanswers[1])
            worksheet.write(row+1,col+9,secanswers[2])
            worksheet.write(row+1,col+10,secanswers[3])
            worksheet.write(row+1,col+11,secanswers[4])
        for i, answerrr in enumerate(trdanswers):
            print("Q{}: {}".format(i + 11, answerrr))
            worksheet.write(row+1,col+12,trdanswers[0])
            worksheet.write(row+1,col+13,trdanswers[1])
        for i, answerrrr in enumerate(fouanswers):
            print("Q{}: {}".format(i + 13, answerrrr))
            worksheet.write(row+1,col+14,fouanswers[0])
            worksheet.write(row+1,col+15,fouanswers[1])
        for i, answerrrrr in enumerate(fivanswers):
            print("Q{}: {}".format(i + 15, answerrrrr))
            worksheet.write(row+1,col+16,fivanswers[0])
            worksheet.write(row+1,col+17,fivanswers[1])
        for i, answerrrrrr in enumerate(sixanswers):
            print("Q{}: {}".format(i + 17, answerrrrrr))
            worksheet.write(row+1,col+18,sixanswers[0])
            worksheet.write(row+1,col+19,sixanswers[1])
        for i, answerrrrrrr in enumerate(sevanswers):
            print("Q{}: {}".format(i + 19, answerrrrrrr))
            worksheet.write(row+1,col+20,sevanswers[0])
            worksheet.write(row+1,col+21,sevanswers[1])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(im,erp,(570,55),font,.5,(0,0,255),1,cv2.LINE_AA)
        cv2.putText(im,testid,(580,95),font,.5,(0,0,255),1,cv2.LINE_AA)
        #cv2.putText(im,answers[0],(500,200),font,.5,(0,0,255),1,cv2.LINE_AA)
        #cv2.putText(im,answers[1],(500,220),font,.5,(0,0,255),1,cv2.LINE_AA)
        cv2.imwrite("K:\omr-master\output\{1}_{0}.jpg".format(erp,testid), im)
        cv2.imshow('trans', im)
        cv2.waitKey(0)
        cv2.destroyAllWindows
        row +=1
    workbook.close()
        #print("Close image window and hit ^C to quit.")
        #while True:
        #cv2.waitKey()
if __name__ == '__main__':
        main()