import cv2

# 좌표를 저장할 리스트 초기화
points = []

def click_event(event, x, y, flags, param):
    # 마우스 왼쪽 버튼 클릭 이벤트가 발생하면 실행
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))  # 클릭된 위치의 좌표를 리스트에 추가

        # 선택된 점을 표시
        cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
        cv2.imshow('image', img)
        
        # 4개의 좌표가 모두 선택되었는지 확인
        if len(points) == 4:
            print("Selected Points: ", points)

# 이미지를 불러옴
img = cv2.imread('future_hall_4f.png')
cv2.imshow('image', img)

# 마우스 클릭 이벤트와 click_event 함수를 바인딩
cv2.setMouseCallback('image', click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()


