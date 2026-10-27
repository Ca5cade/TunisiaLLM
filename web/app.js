/**
 * TunisiaLLM - Web UI Frontend Client
 * Handles real-time SSE token streaming, tool call extraction, RTL auto-detection,
 * and reactive UI updates.
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const messagesContainer = document.getElementById("messages-container");
    const welcomeHero = document.getElementById("welcome-hero");
    const chatInput = document.getElementById("chat-input");
    const btnSend = document.getElementById("btn-send");
    const btnClearChat = document.getElementById("btn-clear-chat");
    const btnToggleSettings = document.getElementById("btn-toggle-settings");
    const btnCloseSettings = document.getElementById("btn-close-settings");
    const btnResetSettings = document.getElementById("btn-reset-settings");
    const settingsDrawer = document.getElementById("settings-drawer");
    const drawerOverlay = document.getElementById("drawer-overlay");
    const starterCards = document.querySelectorAll(".starter-card");
    const chipToggles = document.querySelectorAll(".chip-toggle");
    const dockMetrics = document.getElementById("dock-metrics");
    const statSpeed = document.getElementById("stat-speed");
    const statTokens = document.getElementById("stat-tokens");
    const modelStatusText = document.getElementById("model-status-text");
    const hwDevice = document.getElementById("hw-device");
    const hwModelPreset = document.getElementById("hw-model-preset");

    // Setting Inputs
    const settingSystemPrompt = document.getElementById("setting-system-prompt");
    const settingTemp = document.getElementById("setting-temperature");
    const settingTopP = document.getElementById("setting-top-p");
    const settingMaxTokens = document.getElementById("setting-max-tokens");
    const settingRepPenalty = document.getElementById("setting-rep-penalty");
    const valTemp = document.getElementById("val-temperature");
    const valTopP = document.getElementById("val-top-p");
    const valMaxTokens = document.getElementById("val-max-tokens");
    const valRepPenalty = document.getElementById("val-rep-penalty");

    // State
    let isGenerating = false;
    let abortController = null;
    let chatHistory = [];

    // =========================================================================
    // Settings Binding & Persistence
    // =========================================================================
    function loadSavedSettings() {
        const saved = JSON.parse(localStorage.getItem("tunisiallm_settings") || "{}");
        if (saved.systemPrompt) settingSystemPrompt.value = saved.systemPrompt;
        if (saved.temperature) {
            settingTemp.value = saved.temperature;
            valTemp.textContent = saved.temperature;
        }
        if (saved.topP) {
            settingTopP.value = saved.topP;
            valTopP.textContent = saved.topP;
        }
        if (saved.maxTokens) {
            settingMaxTokens.value = saved.maxTokens;
            valMaxTokens.textContent = saved.maxTokens;
        }
        if (saved.repPenalty) {
            settingRepPenalty.value = saved.repPenalty;
            valRepPenalty.textContent = saved.repPenalty;
        }
    }

    function saveSettings() {
        const settings = {
            systemPrompt: settingSystemPrompt.value,
            temperature: settingTemp.value,
            topP: settingTopP.value,
            maxTokens: settingMaxTokens.value,
            repPenalty: settingRepPenalty.value,
        };
        localStorage.setItem("tunisiallm_settings", JSON.stringify(settings));
    }

    settingTemp.addEventListener("input", (e) => {
        valTemp.textContent = e.target.value;
        saveSettings();
    });
    settingTopP.addEventListener("input", (e) => {
        valTopP.textContent = e.target.value;
        saveSettings();
    });
    settingMaxTokens.addEventListener("input", (e) => {
        valMaxTokens.textContent = e.target.value;
        saveSettings();
    });
    settingRepPenalty.addEventListener("input", (e) => {
        valRepPenalty.textContent = e.target.value;
        saveSettings();
    });
    settingSystemPrompt.addEventListener("input", saveSettings);

    btnResetSettings.addEventListener("click", () => {
        settingSystemPrompt.value = "إنت مساعد ذكي بالدارجة التونسية. جاوب بكل طلاقة ووضوح.";
        settingTemp.value = "0.7";
        valTemp.textContent = "0.7";
        settingTopP.value = "0.9";
        valTopP.textContent = "0.9";
        settingMaxTokens.value = "256";
        valMaxTokens.textContent = "256";
        settingRepPenalty.value = "1.15";
        valRepPenalty.textContent = "1.15";
        saveSettings();
    });

    // =========================================================================
    // Drawer Management
    // =========================================================================
    function toggleDrawer(open) {
        settingsDrawer.classList.toggle("active", open);
        drawerOverlay.classList.toggle("active", open);
    }

    btnToggleSettings.addEventListener("click", () => toggleDrawer(true));
    btnCloseSettings.addEventListener("click", () => toggleDrawer(false));
    drawerOverlay.addEventListener("click", () => toggleDrawer(false));

    // Mode chips
    chipToggles.forEach(chip => {
        chip.addEventListener("click", () => {
            chipToggles.forEach(c => c.classList.remove("active"));
            chip.classList.add("active");
            const newPrompt = chip.getAttribute("data-system");
            if (newPrompt) {
                settingSystemPrompt.value = newPrompt;
                saveSettings();
            }
        });
    });

    // =========================================================================
    // Auto-Resize Textarea
    // =========================================================================
    function resizeTextarea() {
        chatInput.style.height = "auto";
        chatInput.style.height = Math.min(chatInput.scrollHeight, 160) + "px";
    }

    chatInput.addEventListener("input", resizeTextarea);
    chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    btnSend.addEventListener("click", (e) => {
        e.preventDefault();
        sendMessage();
    });

    // =========================================================================
    // Starter Prompts Click
    // =========================================================================
    starterCards.forEach(card => {
        card.addEventListener("click", () => {
            const prompt = card.getAttribute("data-prompt");
            if (prompt && !isGenerating) {
                chatInput.value = prompt;
                resizeTextarea();
                sendMessage();
            }
        });
    });

    // =========================================================================
    // Text Direction Detection (RTL vs LTR)
    // =========================================================================
    function isArabicText(text) {
        const arabicRegex = /[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]/;
        return arabicRegex.test(text);
    }

    // =========================================================================
    // Message Rendering & Tool Calls
    // =========================================================================
    function appendMessageRow(role, content = "") {
        if (welcomeHero) welcomeHero.style.display = "none";

        const row = document.createElement("div");
        row.className = `message-row ${role}`;

        const avatar = document.createElement("div");
        avatar.className = "message-avatar";
        avatar.textContent = role === "user" ? "👤" : "🇹🇳";

        const bubble = document.createElement("div");
        bubble.className = "message-bubble";
        bubble.dir = isArabicText(content) ? "rtl" : "ltr";

        const contentDiv = document.createElement("div");
        contentDiv.className = "bubble-content";
        contentDiv.innerHTML = formatMessageContent(content);

        bubble.appendChild(contentDiv);
        row.appendChild(avatar);
        row.appendChild(bubble);

        messagesContainer.appendChild(row);
        scrollToBottom();

        return { row, bubble, contentDiv };
    }

    function formatMessageContent(rawText) {
        if (!rawText) return "";

        // Escape HTML
        let sanitized = rawText
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");

        // Parse Tool Calls: <tool_call>...</tool_call> and results
        const toolCallRegex = /&lt;tool_call&gt;([\s\S]*?)&lt;\/tool_call&gt;/g;
        sanitized = sanitized.replace(toolCallRegex, (match, toolCall) => {
            return `
            <div class="tool-call-block">
                <div class="tool-call-header">
                    <span>⚡ تنفيذ أداة الحساب / البحث (Tool Execution)</span>
                    <span>🔧 Tool Call</span>
                </div>
                <div class="tool-call-body">
                    <div class="tool-call-input"><span class="k">&gt; </span>${toolCall.trim()}</div>
                </div>
            </div>`;
        });

        const toolResultRegex = /&lt;tool_result&gt;([\s\S]*?)&lt;\/tool_result&gt;/g;
        sanitized = sanitized.replace(toolResultRegex, (match, toolResult) => {
            return `
            <div class="tool-call-block">
                <div class="tool-call-body">
                    <div class="tool-call-result">✓ نتيجة الأداة: ${toolResult.trim()}</div>
                </div>
            </div>`;
        });

        // Basic formatting (bold, newlines)
        sanitized = sanitized.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
        sanitized = sanitized.replace(/\n/g, "<br>");

        return sanitized;
    }

    function scrollToBottom() {
        const viewport = document.querySelector(".chat-viewport");
        viewport.scrollTop = viewport.scrollHeight;
    }

    // =========================================================================
    // Core Send & Streaming Logic (SSE)
    // =========================================================================
    let lastSendTimestamp = 0;

    async function sendMessage() {
        const now = performance.now();
        if (isGenerating) {
            // Protect against rapid double clicks: only cancel if > 500ms since generation started
            if (now - lastSendTimestamp < 500) {
                return;
            }
            if (abortController) {
                abortController.abort();
            }
            return;
        }

        lastSendTimestamp = now;
        const text = chatInput.value.trim();
        if (!text) return;

        chatInput.value = "";
        resizeTextarea();

        // 1. Render User Message
        appendMessageRow("user", text);
        chatHistory.push({ role: "user", content: text });

        // 2. Prepare Assistant Bubble
        setGeneratingState(true);
        const { bubble, contentDiv } = appendMessageRow("assistant", "");
        
        // Add animated typing cursor
        const cursor = document.createElement("span");
        cursor.className = "typing-cursor";
        bubble.appendChild(cursor);

        // Track metrics
        const startTime = performance.now();
        let tokenCount = 0;
        let assistantFullText = "";

        abortController = new AbortController();

        try {
            const payload = {
                prompt: text,
                system_prompt: settingSystemPrompt.value,
                temperature: parseFloat(settingTemp.value),
                top_p: parseFloat(settingTopP.value),
                max_new_tokens: parseInt(settingMaxTokens.value),
                repetition_penalty: parseFloat(settingRepPenalty.value),
                chat_history: chatHistory.slice(-6)
            };

            const response = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
                signal: abortController.signal
            });

            if (!response.ok) {
                throw new Error(`Server returned HTTP ${response.status}: ${response.statusText}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let buffer = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n");
                buffer = lines.pop(); // keep remainder

                for (const line of lines) {
                    if (line.startsWith("data: ")) {
                        const dataStr = line.slice(6).trim();
                        if (dataStr === "[DONE]") {
                            break;
                        }
                        try {
                            const parsed = JSON.parse(dataStr);
                            if (parsed.token) {
                                tokenCount++;
                                assistantFullText += parsed.token;
                                contentDiv.innerHTML = formatMessageContent(assistantFullText);
                                bubble.dir = isArabicText(assistantFullText) ? "rtl" : "ltr";
                                
                                // Update live metrics
                                const elapsedSecs = (performance.now() - startTime) / 1000;
                                const tps = (tokenCount / Math.max(elapsedSecs, 0.01)).toFixed(1);
                                statSpeed.textContent = `⚡ ${tps} tok/s`;
                                statTokens.textContent = `${tokenCount} tokens`;
                                dockMetrics.style.display = "flex";

                                scrollToBottom();
                            }
                        } catch (err) {
                            // ignore malformed SSE chunks
                        }
                    }
                }
            }

        } catch (err) {
            if (err.name === "AbortError") {
                assistantFullText += " [تم إيقاف التوليد]";
            } else {
                assistantFullText += `\n[خطأ: ${err.message}]`;
            }
            contentDiv.innerHTML = formatMessageContent(assistantFullText);
        } finally {
            // Remove cursor
            if (cursor && cursor.parentNode) {
                cursor.remove();
            }

            // Append metadata / metrics badge to message bubble
            const elapsedTotal = ((performance.now() - startTime) / 1000).toFixed(2);
            const finalTps = (tokenCount / Math.max(parseFloat(elapsedTotal), 0.01)).toFixed(1);

            const metaBar = document.createElement("div");
            metaBar.className = "message-meta";
            metaBar.innerHTML = `
                <span>⚡ ${finalTps} tok/s</span>
                <span>•</span>
                <span>${tokenCount} tokens in ${elapsedTotal}s</span>
                <button class="btn-copy-msg" title="نسخ">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                    </svg>
                    <span>نسخ</span>
                </button>
            `;

            const copyBtn = metaBar.querySelector(".btn-copy-msg");
            copyBtn.addEventListener("click", () => {
                navigator.clipboard.writeText(assistantFullText);
                copyBtn.innerHTML = `<span>✓ تم النسخ</span>`;
                setTimeout(() => {
                    copyBtn.innerHTML = `<span>نسخ</span>`;
                }, 2000);
            });

            bubble.appendChild(metaBar);
            chatHistory.push({ role: "assistant", content: assistantFullText });
            setGeneratingState(false);
            scrollToBottom();
        }
    }

    function setGeneratingState(generating) {
        isGenerating = generating;
        const sendIcon = btnSend.querySelector(".send-icon");
        const stopIcon = btnSend.querySelector(".stop-icon");

        if (generating) {
            sendIcon.style.display = "none";
            stopIcon.style.display = "block";
            btnSend.classList.add("btn-stop");
            btnSend.title = "إيقاف التوليد (Stop)";
        } else {
            sendIcon.style.display = "block";
            stopIcon.style.display = "none";
            btnSend.classList.remove("btn-stop");
            btnSend.title = "إرسال الرسالة (Send)";
            abortController = null;
        }
    }

    // =========================================================================
    // Clear Chat
    // =========================================================================
    btnClearChat.addEventListener("click", () => {
        if (confirm("هل ترغب في مسح كل المحادثة والبدء من جديد؟")) {
            messagesContainer.innerHTML = "";
            chatHistory = [];
            if (welcomeHero) welcomeHero.style.display = "flex";
            dockMetrics.style.display = "none";
        }
    });

    // =========================================================================
    // Fetch Model Hardware Status
    // =========================================================================
    async function checkModelStatus() {
        try {
            const res = await fetch("/api/status");
            if (res.ok) {
                const data = await res.json();
                modelStatusText.textContent = `${data.preset || '80M'} • ${data.checkpoint_name || 'Active'} • Ready`;
                if (hwDevice) hwDevice.textContent = `${data.device} (${data.device_name || 'Local'})`;
                if (hwModelPreset) hwModelPreset.textContent = `TunisiaLLM ${data.preset} (${data.parameters || '78M'} params)`;
            }
        } catch (err) {
            modelStatusText.textContent = "TunisiaLLM • Connected";
        }
    }

    // Initialize
    loadSavedSettings();
    checkModelStatus();
    window.sendMessage = sendMessage;
});
