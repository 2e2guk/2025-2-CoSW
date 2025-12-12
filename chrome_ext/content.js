// 중복 실행 방지
if (!window.__lost112ChatInjected) {
  window.__lost112ChatInjected = true;
  injectChatbot();
}

function injectChatbot() {
  console.log("Lost112 챗봇 버튼 생성 시작"); // 디버깅용 로그

  // 1. 스타일 정의
  const style = document.createElement("style");
  // 이미지 URL을 변수로 받아서 CSS에 안전하게 넣기
  const iconUrl = chrome.runtime.getURL("icon.png");
  
  style.textContent = `
    #lost112-chatbot-button {
      position: fixed;
      bottom: 75px;
      right: 30px;
      width: 150px;
      height: 150px;
      border-radius: 50%;
      /* 이미지가 없어도 파란색 원이 보이도록 배경색 지정 */
    //   background-color: #1b4ea3; 
      background-image: url('${iconUrl}');
      background-size: contain;
      background-position: center;
      background-repeat: no-repeat;
      cursor: pointer;
      z-index: 2147483647; /* 웹사이트의 다른 요소보다 무조건 위에 뜨게 설정 */
      transition: transform 0.2s;
    }

    #lost112-chatbot-button:hover {
      transform: scale(1.1);
    }

    #lost112-chatbot-window {
      position: fixed;
      top: 0;
      bottom: 0px;
      right: 30px;
      width: 350px;
      height: 500px;
      background: white;
      border-radius: 12px;
      box-shadow: 0 5px 20px rgba(0,0,0,0.2);
      z-index: 2147483646;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      font-family: 'Malgun Gothic', sans-serif;
      border: 1px solid #ddd;
    }

    #lost112-chatbot-header {
      background: #1b4ea3;
      color: white;
      padding: 15px;
      font-weight: bold;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    #lost112-chatbot-close {
      cursor: pointer;
      font-size: 24px;
      background: none;
      border: none;
      color: white;
      padding: 0;
      line-height: 1;
    }

    #lost112-chatbot-body {
      flex: 1;
      padding: 15px;
      background: #f9f9f9;
      overflow-y: auto;
      font-size: 14px;
    }

    #lost112-chatbot-footer {
      padding: 10px;
      background: white;
      border-top: 1px solid #eee;
      display: flex;
      gap: 8px;
    }

    #lost112-chatbot-input {
      flex: 1;
      padding: 8px 12px;
      border-radius: 20px;
      border: 1px solid #ddd;
      outline: none;
    }

    #lost112-chatbot-send {
      background: #1b4ea3;
      color: white;
      border: none;
      border-radius: 20px;
      padding: 0 15px;
      cursor: pointer;
      font-weight: bold;
    }
  `;
  document.head.appendChild(style);

  // 2. 버튼 요소 생성 및 주입
  const btn = document.createElement("div");
  btn.id = "lost112-chatbot-button";
  // 버튼 안에 텍스트나 이미지를 넣고 싶다면 여기에 추가 (현재는 배경이미지 사용)
  document.body.appendChild(btn);

  // 3. 이벤트 리스너
  btn.addEventListener("click", () => {
    openChatPopup();
  });
}

function openChatPopup() {
  const chatWindowURL = chrome.runtime.getURL("chat_window.html");
  
  // 팝업 창 크기 정의
  const winWidth = 450;
  // 모니터 전체 높이에서 여백을 뺀 크기로 설정하여 꽉 찬 느낌을 줌
  // 100%는 안되지만, 모니터 높이만큼 크게 지정하여 시각적으로 꽉 찬 느낌을 줌
  const winHeight = screen.height - 150; 

  // 팝업 창이 오른쪽 상단 모서리에 붙도록 계산
  // 1. 오른쪽 끝에 붙도록 X 좌표 (left) 계산:
  //    (전체 화면 너비) - (창 너비)
  const leftPosition = screen.width - winWidth; 
  
  // 2. 상단 끝에 붙도록 Y 좌표 (top) 계산:
  const topPosition = 0; 

  // 팝업 창의 옵션 설정
  const windowFeatures = 
    `width=${winWidth},height=${winHeight},resizable=yes,scrollbars=no,status=no,` +
    `left=${leftPosition},top=${topPosition}`; 

  // 새 팝업 창을 엽니다.
  window.open(chatWindowURL, "lost112ChatPopup", windowFeatures);
}