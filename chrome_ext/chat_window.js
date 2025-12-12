document.addEventListener('DOMContentLoaded', () => {

    /* ============================================================
       기본 DOM 요소
    ============================================================ */
    const messages = document.getElementById("chatbot-messages");
    const chatInput = document.getElementById("chat-input");
    const chatSend = document.getElementById("chat-send");

    const attachButton = document.getElementById("attach-button");
    const imageInput = document.getElementById("image-input");
    const imagePreviewBox = document.getElementById("image-preview-box");

    let selectedImageFile = null;


    /* ============================================================
       EMAIL LOGIN / REGISTER
    ============================================================ */

    const LOGIN_ENDPOINT = 'http://127.0.0.1:8000/auth/login';
    const REGISTER_ENDPOINT = 'http://127.0.0.1:8000/auth/register';

    function appendBotMessage(text) {
        const wrapper = document.createElement("div");
        wrapper.className = "bot-message-wrapper";

        wrapper.innerHTML = `
            <img class="message-icon" src="./profile.png">
            <div class="message-bubble">${text}</div>
        `;
        messages.appendChild(wrapper);
        messages.scrollTop = messages.scrollHeight;
    }

    function appendUserMessage(text) {
        const wrapper = document.createElement("div");
        wrapper.className = "user-message-wrapper";
        wrapper.innerHTML = `<div class="user-bubble">${text}</div>`;
        messages.appendChild(wrapper);
        messages.scrollTop = messages.scrollHeight;
    }

    function appendEmailInputBubble() {
        const wrapper = document.createElement("div");
        wrapper.className = "bot-message-wrapper";

        wrapper.innerHTML = `
            <img class="message-icon" src="./profile.png">
            <div class="message-bubble" style="display:flex; flex-direction:column; gap:8px;">
                <div>안녕하세요! LOST112 챗봇입니다.<br>이메일을 입력해주세요!</div>

                <input id="email-input-field" type="email" placeholder="you@example.com"
                       style="width: 85%; padding: 10px; border-radius: 10px; border: 1px solid #aaa; font-size: 14px;" />

                <button id="email-input-submit"
                    style="background:#1b4ea3; color:white; border:none; padding:10px; border-radius:10px; cursor:pointer; font-weight:bold;">
                    제출
                </button>

                <div id="email-input-error" style="color:red; font-size:13px; display:none;"></div>
            </div>
        `;
        messages.appendChild(wrapper);
        messages.scrollTop = messages.scrollHeight;

        document.getElementById("email-input-submit").addEventListener("click", handleEmailSubmit);
        document.getElementById("email-input-field").addEventListener("keydown", e => {
            if (e.key === "Enter") handleEmailSubmit();
        });
    }

    function validateEmail(email) {
        return /\S+@\S+\.\S+/.test(email);
    }

    async function handleEmailSubmit() {
        const emailField = document.getElementById("email-input-field");
        const errorField = document.getElementById("email-input-error");
        const email = emailField.value.trim();

        if (!validateEmail(email)) {
            errorField.innerText = "유효한 이메일을 입력해주세요.";
            errorField.style.display = "block";
            return;
        }
        errorField.style.display = "none";

        try {
            let res = await fetch(LOGIN_ENDPOINT, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ email })
            });

            if (res.ok) {
                const data = await res.json();
                saveAuth(data);
                appendBotMessage("로그인되었습니다! 무엇을 도와드릴까요?");
                finishEmailInput();
                return;
            }

            res = await fetch(REGISTER_ENDPOINT, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ email })
            });

            if (res.ok) {
                res = await fetch(LOGIN_ENDPOINT, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ email })
                });

                const data = await res.json();
                saveAuth(data);
                appendBotMessage("회원가입 완료! 로그인되었습니다.");
                finishEmailInput();
                return;
            }

            errorField.innerText = "회원가입 실패했습니다.";
            errorField.style.display = "block";

        } catch (e) {
            console.error(e);
            errorField.innerText = "서버 오류가 발생했습니다.";
            errorField.style.display = "block";
        }
    }

    function saveAuth(data) {
        localStorage.setItem("access_token", data.access_token);
        localStorage.setItem("user_email", data.email);
        localStorage.setItem("user_id", data.user_id);
    }

    function finishEmailInput() {
        chatInput.disabled = false;
        chatSend.disabled = false;
    }

    const tokenNow = localStorage.getItem("access_token");
    if (!tokenNow) {
        chatInput.disabled = true;
        chatSend.disabled = true;
        appendEmailInputBubble();
    } else {
        appendBotMessage("다시 오셨군요! 무엇을 도와드릴까요?");
    }


    /* ============================================================
       📸 이미지 첨부 기능
    ============================================================ */

    attachButton.addEventListener("click", () => imageInput.click());

    imageInput.addEventListener("change", () => {
        if (imageInput.files && imageInput.files[0]) {
            selectedImageFile = imageInput.files[0];

            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreviewBox.innerHTML = `
                    <img src="${e.target.result}" style="height:60px; border-radius:8px; border:1px solid #ccc;">
                    <span id="image-preview-remove" style="font-size:20px; font-weight:bold; cursor:pointer;">✖</span>
                `;
                imagePreviewBox.style.display = "flex";
            };
            reader.readAsDataURL(selectedImageFile);
        }
    });

    imagePreviewBox.addEventListener("click", (e) => {
        if (e.target.id === "image-preview-remove") {
            selectedImageFile = null;
            imageInput.value = "";
            imagePreviewBox.style.display = "none";
        }
    });


    /* ============================================================
       ✉️ 메시지 전송
    ============================================================ */

    function sendMessage() {
        const text = chatInput.value.trim();
        if (!text && !selectedImageFile) return;

        if (text) appendUserMessage(text);
        if (selectedImageFile) appendUserImage(selectedImageFile);

        handleUserLostItem(text, selectedImageFile);

        chatInput.value = "";
        selectedImageFile = null;
        imageInput.value = "";
        imagePreviewBox.style.display = "none";
    }

    chatSend.addEventListener("click", sendMessage);
    chatInput.addEventListener("keyup", e => {
        if (e.key === "Enter") sendMessage();
    });

    function appendUserImage(file) {
        const wrapper = document.createElement("div");
        wrapper.className = "user-message-wrapper";

        const imgURL = URL.createObjectURL(file);

        wrapper.innerHTML = `
            <div class="user-bubble">
                <img src="${imgURL}" style="max-width:200px; border-radius:12px; object-fit:cover;">
            </div>
        `;
        messages.appendChild(wrapper);
        messages.scrollTop = messages.scrollHeight;
    }


    /* ============================================================
       🔥 POST /items
    ============================================================ */
    async function handleUserLostItem(text, imageFile) {
        const token = localStorage.getItem("access_token");
        if (!token) return appendBotMessage("로그인 후 이용해주세요.");

        const formData = new FormData();
        if (text) formData.append("description", text);
        if (imageFile) formData.append("image", imageFile);

        try {
            const res = await fetch("http://127.0.0.1:8000/items", {
                method: "POST",
                headers: { "Authorization": `Bearer ${token}` },
                body: formData
            });

            const data = await res.json();
            appendMatchResult(data);

        } catch (err) {
            console.error(err);
            appendBotMessage("서버 오류가 발생했습니다.");
        }
    }


    /* ============================================================
       🎯 매칭 결과 UI
    ============================================================ */
    function appendMatchResult(data) {
        const wrapper = document.createElement("div");
        wrapper.className = "bot-message-wrapper";

        let html = `
            <img class="message-icon" src="./profile.png">
            <div class="message-bubble">
                가장 유사한 분실물 3건을 찾았어요.<br>
                아래에서 확인해주세요.<br><br>
                <div class="match-container">
        `;

        data.immediate_matches.forEach(item => {
            // BE 제공 스키마에 맞춰 사용: title, image, place, detail_link
            const imgUrl = item.image || null;
            const name = item.title || "(항목명 없음)";
            const desc = item.place || "";
            const date = item.date ? ` · 접수일: ${escapeHtml(item.date)}` : "";

            html += `
                <div class="match-card" data-detail="${item.detail_link}">
                    ${imgUrl ? `<img src="${imgUrl}" class="match-img">` : ``}
                    <div class="match-name">${escapeHtml(name)}</div>
                    <div class="match-desc">${escapeHtml(desc)}${date}</div>
                </div>
            `;
        });

        html += `
                </div>
                <div class="match-actions">
                    <button class="btn-outline no-item-button">제 물건이 없어요 (알림 받기)</button>
                    <button class="btn-primary found-button" data-request-id="${data.registered_item.request_id}">찾았어요 (알림 중지)</button>
                </div>
            </div>
        `;

        wrapper.innerHTML = html;
        messages.appendChild(wrapper);
        messages.scrollTop = messages.scrollHeight;
    }


    /* ============================================================
       이미지 URL 변환 함수
    ============================================================ */
    function resolveImageUrl(url) {
        if (!url) return null;

        // 절대 URL 이면 그대로 사용
        if (url.startsWith("http")) return url;

        // 백엔드가 "/uploads/파일명.png" 처럼 상대 경로를 준 경우
        // 한글/공백/특수문자 포함 파일명을 안전하게 인코딩
        try {
            const encodedPath = encodeURI(url);
            return `http://127.0.0.1:8000${encodedPath}`;
        } catch {
            return `http://127.0.0.1:8000${url}`;
        }
    }

    // 안전한 이미지 URL 반환 (없으면 기본 이미지)
    function safeImage(url) {
        const resolved = resolveImageUrl(url);
        return resolved || "./profile.png";
    }


    /* ============================================================
       📋 사용자 분실물 목록 조회
    ============================================================ */
    async function fetchUserItemsList() {
        const token = localStorage.getItem("access_token");

        try {
            const res = await fetch("http://127.0.0.1:8000/items/list", {
                method: "GET",
                headers: { "Authorization": `Bearer ${token}` }
            });

            const data = await res.json();
            renderUserItemsList(data.items || []);

        } catch (err) {
            console.error(err);
            appendBotMessage("서버 오류가 발생했습니다.");
        }
    }


    /* ============================================================
       목록 렌더링 (match-card → user-item-card로 분리)
    ============================================================ */
    function renderUserItemsList(items) {
        const wrapper = document.createElement("div");
        wrapper.className = "bot-message-wrapper";

        let html = `
            <img class="message-icon" src="./profile.png">
            <div class="message-bubble">
                나의 분실물 목록 (${items.length}개)<br><br>
                <div class="match-container">
        `;

        if (!items.length) {
            html += `등록된 분실물이 없습니다.`;
        } else {
            items.forEach(item => {

                const imgUrl = resolveImageUrl(item.original_image_url);

                // 상태 텍스트: 항상 진행중으로 표시
                const statusText = "진행중";

                html += `
                    <div class="user-item-card" data-request-id="${item.request_id}">
                        ${imgUrl ? `<img src="${imgUrl}" class="match-img">` : `<div class="user-item-placeholder">LOST112</div>`}

                        <div class="match-name">${escapeHtml(item.description || "(설명 없음)")}</div>
                        <div class="match-desc">
                            상태: ${statusText} · 등록일: ${formatDate(item.created_at)}
                        </div>
                `;

                // 진행중일 때만 알림 중지 버튼 표시
                html += `
                        <button class="cease-button" data-request-id="${item.request_id}">
                            알림 중지
                        </button>
                `;

                html += `</div>`;
            });
        }

        html += `
                </div>
            </div>
        `;

        wrapper.innerHTML = html;
        messages.appendChild(wrapper);
        messages.scrollTop = messages.scrollHeight;
    }



    /* ============================================================
       헬퍼 함수
    ============================================================ */
    function escapeHtml(str) {
        return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    function formatDate(iso) {
        if (!iso) return "-";
        try {
            const d = new Date(iso);
            return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
        } catch {
            return iso;
        }
    }
    

    function formatDate(iso) {
        if (!iso) return "-";
        try {
            const d = new Date(iso);
            return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
        } catch {
            return iso;
        }
    }
    function renderHelpCard() {
        const wrapper = document.createElement("div");
        wrapper.className = "bot-message-wrapper help-anim-wrapper";

        wrapper.innerHTML = `
            <img class="message-icon" src="./profile.png">
            <div class="help-card slide-in">

                <div class="help-title">🧭 도움말 안내</div>

                <div class="help-item">
                    <span class="help-icon">🔍</span>
                    <div class="help-text">
                        <div class="help-head">분실물 찾기</div>
                        <div class="help-desc">물건 설명 또는 사진을 보내면 AI가 가장 유사한 습득물을 자동으로 찾아드립니다.</div>
                    </div>
                </div>

                <div class="help-item">
                    <span class="help-icon">📁</span>
                    <div class="help-text">
                        <div class="help-head">내 분실물 목록</div>
                        <div class="help-desc">내가 등록한 분실물 기록 확인 및 알림 상태를 관리할 수 있어요.</div>
                    </div>
                </div>

                <div class="help-item">
                    <span class="help-icon">📸</span>
                    <div class="help-text">
                        <div class="help-head">사진 첨부</div>
                        <div class="help-desc">사진을 함께 보내면 AI가 색상·형태를 분석해 더 정확한 매칭을 제공합니다.</div>
                    </div>
                </div>

                <div class="help-item">
                    <span class="help-icon">🔔</span>
                    <div class="help-text">
                        <div class="help-head">알림 기능</div>
                        <div class="help-desc">유사한 습득물이 접수되면 자동으로 이메일로 알려드립니다.</div>
                    </div>
                </div>

            </div>
        `;

        messages.appendChild(wrapper);
        messages.scrollTop = messages.scrollHeight;
    }





    /* ============================================================
       클릭 이벤트 (목록 카드 / 매칭 카드 분리)
    ============================================================ */
    document.addEventListener("click", (e) => {
        // 버튼/액션 우선 처리
        if (e.target.classList.contains("cease-button")) {
            stopNotification(e.target.dataset.requestId);
            return;
        }
        if (e.target.classList.contains("list-button")) {
            fetchUserItemsList();
            return;
        }
        if (e.target.classList.contains("found-button")) {
            stopNotification(e.target.dataset.requestId);
            return;
        }
        if (e.target.classList.contains("no-item-button")) {
            appendBotMessage("걱정마세요! 유사한 물건이 등록되는 즉시 이메일로 알려드릴게요.");
            return;
        }

        // 🔵 목록 카드 클릭은 아무 동작도 하지 않음 (버튼 제외)
        if (e.target.closest(".user-item-card")) {
            return;
        }

        // 🟡 경찰청 매칭 카드
        if (e.target.closest(".match-card")) {
            const url = e.target.closest(".match-card").dataset.detail;
            if (url) window.open(url, "_blank");
            return;
        }
    });
    const openListBtn = document.getElementById("open-list");
    if (openListBtn) {
        openListBtn.addEventListener("click", () => {
            fetchUserItemsList();
        });
    }

    const openHelpBtn = document.getElementById("open-help");
    if (openHelpBtn) {
        openHelpBtn.addEventListener("click", () => {
            renderHelpCard();
        });
    }

    /* ============================================================
       PATCH /items/cease
    ============================================================ */
    async function stopNotification(requestId) {
        const token = localStorage.getItem("access_token");

        try {
            const res = await fetch(`http://127.0.0.1:8000/items/cease/${requestId}`, {
                method: "PATCH",
                headers: { "Authorization": `Bearer ${token}` }
            });

            const data = await res.json();
            appendBotMessage(data.message || "알림이 중지되었습니다.");

            // UI 즉시 반영: 해당 카드 상태를 '찾음'으로 변경하고 버튼 제거
            const card = document.querySelector(`.user-item-card[data-request-id='${requestId}']`);
            if (card) {
                const descEl = card.querySelector('.match-desc');
                if (descEl) {
                    // 기존 '상태: 진행중 · 등록일: yyyy-mm-dd' 문구에서 상태만 교체
                    const text = descEl.textContent;
                    const replaced = text.replace(/상태:\s*진행중/, '상태: 찾음');
                    descEl.textContent = replaced;
                }
                const ceaseBtn = card.querySelector('.cease-button');
                if (ceaseBtn) ceaseBtn.remove();
            }

        } catch (err) {
            console.error(err);
            appendBotMessage("알림 중지에 실패했습니다.");
        }
    }

});
