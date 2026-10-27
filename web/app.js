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

    // Voice & Audio DOM Elements
    const btnMic = document.getElementById("btn-mic");
    const voiceStatusBar = document.getElementById("voice-status-bar");
    const voiceStatusText = document.getElementById("voice-status-text");
    const btnVoiceCancel = document.getElementById("btn-voice-cancel");
    const btnVoiceToggle = document.getElementById("btn-voice-toggle");
    const settingVoiceLang = document.getElementById("setting-voice-lang");
    const settingAutoSpeak = document.getElementById("setting-auto-speak");
    const settingAutoSend = document.getElementById("setting-auto-send");

    // State
    let isGenerating = false;
    let abortController = null;
    let chatHistory = [];
    
    // Voice State
    let isRecording = false;
    let recognition = null;
    let autoSpeak = false;
    let autoSendSpeech = true;
    let voiceLang = "ar-TN";
    let speechTimer = null;
    let currentlySpeakingBtn = null;

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
        if (saved.voiceLang) {
            voiceLang = saved.voiceLang;
            if (settingVoiceLang) settingVoiceLang.value = saved.voiceLang;
        }
        if (saved.autoSpeak !== undefined) {
            autoSpeak = saved.autoSpeak;
            if (settingAutoSpeak) settingAutoSpeak.checked = saved.autoSpeak;
            updateVoiceToggleButton();
        }
        if (saved.autoSendSpeech !== undefined) {
            autoSendSpeech = saved.autoSendSpeech;
            if (settingAutoSend) settingAutoSend.checked = saved.autoSendSpeech;
        }
    }

    function saveSettings() {
        const settings = {
            systemPrompt: settingSystemPrompt.value,
            temperature: settingTemp.value,
            topP: settingTopP.value,
            maxTokens: settingMaxTokens.value,
            repPenalty: settingRepPenalty.value,
            voiceLang: settingVoiceLang ? settingVoiceLang.value : voiceLang,
            autoSpeak: autoSpeak,
            autoSendSpeech: autoSendSpeech
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

    if (settingVoiceLang) {
        settingVoiceLang.addEventListener("change", (e) => {
            voiceLang = e.target.value;
            saveSettings();
        });
    }

    if (settingAutoSpeak) {
        settingAutoSpeak.addEventListener("change", (e) => {
            autoSpeak = e.target.checked;
            updateVoiceToggleButton();
            saveSettings();
        });
    }

    if (settingAutoSend) {
        settingAutoSend.addEventListener("change", (e) => {
            autoSendSpeech = e.target.checked;
            saveSettings();
        });
    }

    function updateVoiceToggleButton() {
        if (!btnVoiceToggle) return;
        if (autoSpeak) {
            btnVoiceToggle.classList.add("active");
            btnVoiceToggle.textContent = "🔊 الرد الصوتي: مفعّل";
        } else {
            btnVoiceToggle.classList.remove("active");
            btnVoiceToggle.textContent = "🔊 الرد الصوتي: إيقاف";
        }
    }

    if (btnVoiceToggle) {
        btnVoiceToggle.addEventListener("click", () => {
            autoSpeak = !autoSpeak;
            if (settingAutoSpeak) settingAutoSpeak.checked = autoSpeak;
            updateVoiceToggleButton();
            saveSettings();
        });
    }

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
        if (settingVoiceLang) settingVoiceLang.value = "ar-TN";
        if (settingAutoSpeak) settingAutoSpeak.checked = false;
        if (settingAutoSend) settingAutoSend.checked = true;
        voiceLang = "ar-TN";
        autoSpeak = false;
        autoSendSpeech = true;
        updateVoiceToggleButton();
        saveSettings();
    });

    // =========================================================================
    // Speech Recognition (Speech-to-Text / Microphone)
    // =========================================================================
    function initSpeechRecognition() {
        if (!btnMic) return;

        const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRec) {
            btnMic.title = "المتصفح لا يدعم التعرف على الصوت (Web Speech API غير مدعوم)";
            btnMic.addEventListener("click", () => {
                alert("عذراً، متصفحك لا يدعم التعرف الصوتي (Web Speech Recognition). يرجى استعمال متصفح Google Chrome أو Microsoft Edge.");
            });
            return;
        }

        try {
            recognition = new SpeechRec();
            recognition.lang = settingVoiceLang ? settingVoiceLang.value : (voiceLang || "ar-TN");
            recognition.continuous = true;
            recognition.interimResults = true;

            recognition.onstart = () => {
                isRecording = true;
                btnMic.classList.add("recording");
                btnMic.title = "إيقاف التسجيل وإرسال (Stop Recording)";
                if (voiceStatusBar) voiceStatusBar.style.display = "flex";
                if (voiceStatusText) voiceStatusText.textContent = "🎙️ نسمع فيك... تفضل تكلم بالدارجة التونسية";
            };

            recognition.onresult = (event) => {
                let fullTranscript = "";
                for (let i = 0; i < event.results.length; i++) {
                    fullTranscript += event.results[i][0].transcript + " ";
                }
                const cleanText = fullTranscript.trim();
                if (cleanText) {
                    chatInput.value = cleanText;
                    resizeTextarea();
                    chatInput.dir = isArabicText(cleanText) ? "rtl" : "ltr";
                }

                // If auto-send on silence is enabled
                if (autoSendSpeech) {
                    clearTimeout(speechTimer);
                    speechTimer = setTimeout(() => {
                        if (isRecording && chatInput.value.trim().length > 0) {
                            stopSpeechRecognition(true);
                        }
                    }, 1400); // 1.4s silence
                }
            };

            recognition.onerror = (event) => {
                console.warn("Speech recognition error:", event.error);
                if (event.error === "not-allowed" || event.error === "service-not-allowed") {
                    alert("يرجى إعطاء الإذن للمتصفح لاستخدام الميكروفون (Microphone Permission Required).");
                }
                stopSpeechRecognition(false);
            };

            recognition.onend = () => {
                isRecording = false;
                btnMic.classList.remove("recording");
                btnMic.title = "تكلم بالميكروفون (Voice Input)";
                if (voiceStatusBar) voiceStatusBar.style.display = "none";
            };

            btnMic.addEventListener("click", () => {
                if (isRecording) {
                    stopSpeechRecognition(true);
                } else {
                    startSpeechRecognition();
                }
            });

            if (btnVoiceCancel) {
                btnVoiceCancel.addEventListener("click", () => {
                    stopSpeechRecognition(false);
                });
            }
        } catch (e) {
            console.error("Failed to initialize speech recognition:", e);
        }
    }

    function startSpeechRecognition() {
        if (!recognition || isGenerating) return;
        try {
            recognition.lang = settingVoiceLang ? settingVoiceLang.value : (voiceLang || "ar-TN");
            recognition.start();
        } catch (e) {
            console.error("Failed to start speech recognition:", e);
        }
    }

    function stopSpeechRecognition(shouldSend = false) {
        if (speechTimer) clearTimeout(speechTimer);
        if (recognition && isRecording) {
            try {
                recognition.stop();
            } catch (e) {}
        }
        isRecording = false;
        if (btnMic) btnMic.classList.remove("recording");
        if (voiceStatusBar) voiceStatusBar.style.display = "none";

        if (shouldSend && chatInput.value.trim().length > 0) {
            setTimeout(() => {
                sendMessage();
            }, 300);
        }
    }

    // =========================================================================
    // Speech Synthesis (Text-to-Speech / Voice Reply)
    // =========================================================================
    function cleanTextForSpeech(rawText) {
        if (!rawText) return "";
        let cleaned = rawText;
        // Strip tool call blocks and results
        cleaned = cleaned.replace(/<tool_call>[\s\S]*?<\/tool_call>/g, "");
        cleaned = cleaned.replace(/<tool_result>[\s\S]*?<\/tool_result>/g, "");
        // Strip markdown formatting & links
        cleaned = cleaned.replace(/[*_#`~>]/g, "");
        cleaned = cleaned.replace(/\[.*?\]\(.*?\)/g, "");
        return cleaned.trim();
    }

    function getBestArabicVoice() {
        if (!window.speechSynthesis) return null;
        const voices = window.speechSynthesis.getVoices();
        if (!voices || voices.length === 0) return null;

        // Try to match Tunisian or Arabic voice
        return voices.find(v => v.lang.toLowerCase().includes("ar-tn")) ||
               voices.find(v => v.lang.toLowerCase().startsWith("ar")) ||
               voices.find(v => v.lang.toLowerCase().startsWith("fr")) ||
               voices[0];
    }

    function getSpeakerIcon(isPlaying) {
        if (isPlaying) {
            return `
                <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
                    <rect x="6" y="6" width="12" height="12" rx="2"/>
                </svg>
                <span>إيقاف</span>
            `;
        }
        return `
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
                <path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path>
                <path d="M19.07 4.93a10 10 0 0 1 0 14.14"></path>
            </svg>
            <span>استماع</span>
        `;
    }

    function speakText(rawText, btnElement = null) {
        if (!window.speechSynthesis) return;

        // If already speaking, cancel
        if (window.speechSynthesis.speaking) {
            window.speechSynthesis.cancel();
            if (currentlySpeakingBtn) {
                currentlySpeakingBtn.classList.remove("playing");
                currentlySpeakingBtn.innerHTML = getSpeakerIcon(false);
                currentlySpeakingBtn = null;
            }
            if (btnElement === currentlySpeakingBtn) {
                return;
            }
        }

        const cleanText = cleanTextForSpeech(rawText);
        if (!cleanText) return;

        const utterance = new SpeechSynthesisUtterance(cleanText);
        const voice = getBestArabicVoice();
        if (voice) {
            utterance.voice = voice;
            utterance.lang = voice.lang;
        } else {
            utterance.lang = "ar-SA";
        }
        utterance.rate = 1.0;
        utterance.pitch = 1.0;

        if (btnElement) {
            currentlySpeakingBtn = btnElement;
            btnElement.classList.add("playing");
            btnElement.innerHTML = getSpeakerIcon(true);
        }

        utterance.onend = () => {
            if (btnElement) {
                btnElement.classList.remove("playing");
                btnElement.innerHTML = getSpeakerIcon(false);
            }
            if (currentlySpeakingBtn === btnElement) currentlySpeakingBtn = null;
        };

        utterance.onerror = () => {
            if (btnElement) {
                btnElement.classList.remove("playing");
                btnElement.innerHTML = getSpeakerIcon(false);
            }
            if (currentlySpeakingBtn === btnElement) currentlySpeakingBtn = null;
        };

        window.speechSynthesis.speak(utterance);
    }

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
        const pairedRegex = /&lt;tool_call&gt;([\s\S]*?)&lt;\/tool_call&gt;[\s\n]*&lt;tool_result&gt;([\s\S]*?)&lt;\/tool_result&gt;/g;
        sanitized = sanitized.replace(pairedRegex, (match, toolCall, toolRes) => {
            return `
            <div class="tool-call-block">
                <div class="tool-call-header">
                    <span>⚡ تنفيذ أداة الحساب / البحث (Tool Execution)</span>
                    <span>🔧 Tool Call</span>
                </div>
                <div class="tool-call-body">
                    <div class="tool-call-input"><span class="k">&gt; </span>${toolCall.trim()}</div>
                    <div class="tool-call-result">✓ نتيجة الأداة: ${toolRes.trim()}</div>
                </div>
            </div>`;
        });

        const singleToolCallRegex = /&lt;tool_call&gt;([\s\S]*?)&lt;\/tool_call&gt;/g;
        sanitized = sanitized.replace(singleToolCallRegex, (match, toolCall) => {
            return `
            <div class="tool-call-block">
                <div class="tool-call-header">
                    <span>⚡ تنفيذ أداة الحساب / البحث (Tool Execution)</span>
                    <span>🔧 Tool Call</span>
                </div>
                <div class="tool-call-body">
                    <div class="tool-call-input"><span class="k">&gt; </span>${toolCall.trim()}</div>
                    <div class="tool-call-result" style="color: #FBBF24;">⏳ جارٍ الحساب...</div>
                </div>
            </div>`;
        });

        const singleToolResultRegex = /&lt;tool_result&gt;([\s\S]*?)&lt;\/tool_result&gt;/g;
        sanitized = sanitized.replace(singleToolResultRegex, (match, toolResult) => {
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
                <button class="btn-copy-msg" title="نسخ النص">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                    </svg>
                    <span>نسخ</span>
                </button>
                <button class="btn-speak-msg" title="استمع للرد (Listen to speech)">
                    ${getSpeakerIcon(false)}
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

            const speakBtn = metaBar.querySelector(".btn-speak-msg");
            speakBtn.addEventListener("click", () => {
                speakText(assistantFullText, speakBtn);
            });

            bubble.appendChild(metaBar);
            chatHistory.push({ role: "assistant", content: assistantFullText });
            setGeneratingState(false);
            scrollToBottom();

            // Auto-speak response if enabled
            if (autoSpeak) {
                speakText(assistantFullText, speakBtn);
            }
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
    initSpeechRecognition();
    if (window.speechSynthesis) {
        window.speechSynthesis.onvoiceschanged = () => {
            getBestArabicVoice();
        };
    }
    window.sendMessage = sendMessage;
});
