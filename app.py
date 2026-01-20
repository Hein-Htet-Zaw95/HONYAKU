import base64
import io
import os
import time
import tempfile
from typing import Literal

import streamlit as st
from dotenv import load_dotenv
from langdetect import detect
from audio_recorder_streamlit import audio_recorder
from openai import OpenAI
from pypdf import PdfReader

# -----------------------------
# 初期化
# -----------------------------
load_dotenv()
if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"].strip():
    st.warning("OPENAI_API_KEY が設定されていません。 .env に追加してください。")

client = OpenAI()

APP_TITLE = "🇻🇳⇄🇯🇵⇄🇺🇸 多言語翻訳 (テキスト + 音声 + 画像/PDF)"
STT_MODEL = "whisper-1"     # 音声→テキスト
TTS_MODEL = "tts-1"         # テキスト→音声
LLM_MODEL = "gpt-4o-mini"   # 翻訳

# Mobile-friendly: wide layout collapses sidebar by default on phones
st.set_page_config(page_title=APP_TITLE, page_icon="🌏", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .mode-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0 0 1rem 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .language-badge {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        margin: 0.2rem;
        font-weight: 500;
    }
    .translation-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    .transcript-box {
        background: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        color: #333;
    }
    .sidebar-section {
        background: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Enhanced Loading Animations */
    .loading-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        text-align: center;
    }
    
    .loading-spinner {
        width: 60px;
        height: 60px;
        border: 4px solid rgba(255, 255, 255, 0.3);
        border-top: 4px solid white;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-bottom: 1rem;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .loading-dots {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        margin-left: 8px;
    }
    
    .loading-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #667eea;
        animation: pulse 1.5s infinite;
    }
    
    .loading-dot:nth-child(2) { animation-delay: 0.3s; }
    .loading-dot:nth-child(3) { animation-delay: 0.6s; }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.3; transform: scale(0.8); }
        50% { opacity: 1; transform: scale(1.2); }
    }
    
    .progress-bar {
        width: 100%;
        height: 6px;
        background: rgba(255, 255, 255, 0.3);
        border-radius: 3px;
        overflow: hidden;
        margin-top: 1rem;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #00f2ff, #ff00ea);
        border-radius: 3px;
        animation: progress 3s ease-in-out infinite;
    }
    
    @keyframes progress {
        0% { width: 0%; }
        50% { width: 70%; }
        100% { width: 100%; }
    }
    
    .ai-thinking {
        font-size: 1.2rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    .thinking-text {
        opacity: 0.8;
        font-size: 0.9rem;
    }
    
    /* Mobile-friendly audio controls */
    audio {
        width: 100% !important;
        max-width: 100% !important;
        height: 40px !important;
    }
    
    /* Mobile responsive adjustments */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.5rem !important;
        }
        .translation-box {
            font-size: 1.2rem !important;
        }
        .mode-section {
            padding: 1rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🌏 多言語AI翻訳アシスタント</h1><p>ベトナム語 ⇄ 日本語 ⇄ 英語 ⇄ ベンガル語 ⇄ インドネシア語翻訳システム</p></div>', unsafe_allow_html=True)

# Keep language choices in session and provide a one-click swap
if "src" not in st.session_state:
    st.session_state.src = "ja"  # Default to Japanese
if "dst" not in st.session_state:
    st.session_state.dst = "vi"  # Default to Vietnamese

def swap_langs():
    st.session_state.src, st.session_state.dst = st.session_state.dst, st.session_state.src

# -----------------------------
# ヘルパー関数
# -----------------------------

def show_loading_animation(title: str, subtitle: str = ""):
    """Display an animated loading screen with progress bar and spinner"""
    st.markdown(f"""
    <div class="loading-container">
        <div class="loading-spinner"></div>
        <div class="ai-thinking">{title}</div>
        <div class="thinking-text">{subtitle}</div>
        <div class="progress-bar">
            <div class="progress-fill"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def show_typing_animation(text: str):
    """Show a typing animation effect for AI responses"""
    st.markdown(f"""
    <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; 
                border-left: 4px solid #667eea; margin: 1rem 0;">
        <span style="color: #667eea; font-weight: 500;">🤖 AI が考えています</span>
        <span class="loading-dots">
            <span class="loading-dot"></span>
            <span class="loading-dot"></span>
            <span class="loading-dot"></span>
        </span>
        <div style="margin-top: 0.5rem; color: #666; font-style: italic;">{text}</div>
    </div>
    """, unsafe_allow_html=True)

def detect_lang_simple(text: str) -> str:
    """ベトナム語/日本語/英語/ベンガル語/インドネシア語の簡易判定"""
    if any("あ" <= ch <= "ん" or "ア" <= ch <= "ン" or "一" <= ch <= "龯" for ch in text):
        return "ja"
    # Check for Bengali script
    if any("ক" <= ch <= "৻" for ch in text):
        return "bn"
    try:
        lang = detect(text)
        if lang in ("ja", "vi", "en", "bn", "id"):
            return lang
    except Exception:
        pass
    # Simple heuristic: if mostly ASCII, likely English, Vietnamese, or Indonesian
    if all(ord(c) < 128 for c in text):
        # Basic check for English vs Vietnamese vs Indonesian
        english_words = ["the", "and", "is", "are", "was", "were", "have", "has", "will", "would", "can", "could"]
        vietnamese_chars = ["ă", "â", "đ", "ê", "ô", "ơ", "ư", "á", "à", "ả", "ã", "ạ"]
        indonesian_words = ["yang", "dan", "ini", "itu", "dengan", "dari", "untuk", "pada", "dalam", "tidak"]
        
        text_lower = text.lower()
        has_english = any(word in text_lower for word in english_words)
        has_vietnamese = any(char in text_lower for char in vietnamese_chars)
        has_indonesian = any(word in text_lower for word in indonesian_words)
        
        if has_vietnamese:
            return "vi"
        elif has_indonesian:
            return "id"
        elif has_english:
            return "en"
        else:
            return "vi"  # default fallback
    return "ja"


def detect_formality_and_context(text: str, lang: str) -> dict:
    """AI-powered formality and context detection"""
    analysis_prompt = f"""
    Analyze the following text in {lang} language and determine:
    1. Formality level: casual, neutral, formal, very_formal
    2. Context: personal, business, academic, technical, creative, medical, legal
    3. Tone: friendly, professional, serious, playful, urgent, polite
    
    Text: "{text}"
    
    Respond with only a JSON object like:
    {{"formality": "formal", "context": "business", "tone": "professional"}}
    """
    
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": analysis_prompt}],
            temperature=0.1,
            max_tokens=100
        )
        
        import json
        result = json.loads(resp.choices[0].message.content.strip())
        return result
    except Exception:
        # Fallback to simple heuristics
        text_lower = text.lower()
        
        # Simple formality detection
        formal_indicators = ["please", "thank you", "sincerely", "respectfully", "でございます", "いたします", "xin chào", "kính chào"]
        casual_indicators = ["hey", "yo", "だよ", "だね", "ね", "よ", "chào bạn", "ơi"]
        
        if any(indicator in text_lower for indicator in formal_indicators):
            formality = "formal"
        elif any(indicator in text_lower for indicator in casual_indicators):
            formality = "casual"
        else:
            formality = "neutral"
            
        return {"formality": formality, "context": "personal", "tone": "friendly"}


def translate_text(text: str, src: str, dst: str) -> str:
    if src == "auto":
        detected = detect_lang_simple(text)
        if detected in ("vi", "ja", "en", "bn", "id"):
            src = detected
        else:
            src = "vi"  # default fallback
    if src == dst:
        return text

    # AI-powered context analysis
    context_info = detect_formality_and_context(text, src)
    formality = context_info.get("formality", "neutral")
    context = context_info.get("context", "personal")
    tone = context_info.get("tone", "friendly")

    # Create context-aware system prompt
    base_prompt = "あなたはプロの翻訳者です。"
    
    if formality == "very_formal":
        style_instruction = "最も丁寧で格式高い表現を使用し、敬語を適切に使い分けてください。"
    elif formality == "formal":
        style_instruction = "丁寧で正式な表現を使用し、ビジネス文書や公式な場面に適した翻訳をしてください。"
    elif formality == "casual":
        style_instruction = "自然でカジュアルな表現を使用し、日常会話に適した親しみやすい翻訳をしてください。"
    else:  # neutral
        style_instruction = "自然で適度に丁寧な表現を使用してください。"
    
    if context == "business":
        context_instruction = "ビジネス文書として適切な専門用語と表現を使用してください。"
    elif context == "academic":
        context_instruction = "学術的で正確な表現を使用し、専門性を保ってください。"
    elif context == "technical":
        context_instruction = "技術的な内容として正確性を重視し、専門用語を適切に翻訳してください。"
    elif context == "medical":
        context_instruction = "医療用語を正確に翻訳し、専門性と正確性を最優先してください。"
    elif context == "legal":
        context_instruction = "法的文書として正確で曖昧さのない表現を使用してください。"
    else:  # personal, creative
        context_instruction = "感情やニュアンスを大切にし、人間味のある自然な表現を心がけてください。"

    system_prompt = f"""
    {base_prompt}
    
    翻訳スタイル: {style_instruction}
    文脈考慮: {context_instruction}
    
    - ソース言語: 'vi'=ベトナム語, 'ja'=日本語, 'en'=英語, 'bn'=ベンガル語, 'id'=インドネシア語
    - ターゲット言語: 'ja'=日本語, 'vi'=ベトナム語, 'en'=英語, 'bn'=ベンガル語, 'id'=インドネシア語
    - 検出された調子: {tone}
    - 文脈: {context}
    - 丁寧度: {formality}
    
    元のテキストの調子と文脈を保ちながら、上記スタイルで翻訳してください。
    数字や名前はそのまま保持し、説明は追加せず翻訳文のみ出力してください。
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"[SRC={src}] [DST={dst}]\n{text}"},
    ]

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip() if resp.choices[0].message.content else "Translation failed"
    except Exception as e:
        return f"Translation error: {str(e)}"


def transcribe_bytes(wav_bytes: bytes, lang_hint: str = "auto") -> str:
    # Validate audio data
    if not wav_bytes or len(wav_bytes) < 1000:  # Too small to be valid audio
        return ""
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(wav_bytes)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as f:
            # Check file size again after writing
            file_size = os.path.getsize(tmp_path)
            if file_size < 1000:  # Too small to be valid audio
                return ""
                
            kwargs = {"model": STT_MODEL, "file": f}
            if lang_hint in ("vi", "ja", "en"):
                kwargs["language"] = lang_hint
            stt = client.audio.transcriptions.create(**kwargs)
        return stt.text.strip() if stt.text else ""
    except Exception as e:
        # Handle OpenAI API errors gracefully
        if "BadRequestError" in str(type(e)) or "bad request" in str(e).lower():
            st.warning("⚠️ 音声データが無効です。もう一度録音してください。")
            return ""
        else:
            st.error(f"音声認識エラー: {str(e)}")
            return ""
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def encode_image_data_url(image_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def extract_text_from_image(image_bytes: bytes, mime_type: str) -> str:
    if not image_bytes:
        return ""
    prompt = (
        "You are an OCR assistant. Extract all readable text from the image. "
        "Return only the text without additional commentary."
    )
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": encode_image_data_url(image_bytes, mime_type)}},
                    ],
                }
            ],
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip() if resp.choices[0].message.content else ""
    except Exception as e:
        st.error(f"画像OCRエラー: {e}")
        return ""


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    if not pdf_bytes:
        return ""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages_text = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages_text.append(page_text)
        return "\n".join(pages_text).strip()
    except Exception as e:
        st.error(f"PDF読み取りエラー: {e}")
        return ""


def speak(text: str, voice: str = "alloy", fmt: str = "wav"):
    """TTS -> bytes with WAV audio output for better mobile compatibility."""
    if not text.strip():
        return b"", "audio/wav"

    try:
        # Always use WAV format for better mobile compatibility
        resp = client.audio.speech.create(
            model=TTS_MODEL,
            voice=voice,
            input=text,
            response_format="wav"
        )
        mime = "audio/wav"
            
        audio_bytes = resp.read()
        return audio_bytes, mime
    except Exception as e:
        st.error(f"TTS error: {e}")
        return b"", "audio/wav"


def play_audio_mobile_friendly(audio_bytes: bytes, mime: str):
    """Play audio with mobile browser compatibility."""
    if not audio_bytes:
        return
        
    # Use autoplay=False and start_time=0 for better mobile compatibility
    try:
        st.audio(audio_bytes, format=mime, start_time=0, autoplay=False)
        
        # Add a mobile-friendly play button instruction
        st.markdown("""
        <div style="text-align: center; margin: 0.5rem 0; padding: 0.5rem; 
                    background: #e8f4fd; border-radius: 8px; color: #1f77b4;">
            📱 <strong>スマートフォンユーザーへ:</strong> 
            上の再生ボタンをタップして音声を聞いてください。<br>
            <em>For mobile users: Tap the play button above to hear the audio.</em>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.warning(f"音声再生エラー: {e}")
        
        # Fallback: Provide download link for mobile
        import base64
        b64_audio = base64.b64encode(audio_bytes).decode()
        audio_html = f"""
        <div style="text-align: center; margin: 1rem 0;">
            <a href="data:{mime};base64,{b64_audio}" download="translation_audio.wav" 
               style="background: #667eea; color: white; padding: 0.5rem 1rem; 
                      border-radius: 20px; text-decoration: none;">
                📱 音声をダウンロード / Download Audio (WAV)
            </a>
        </div>
        """
        st.markdown(audio_html, unsafe_allow_html=True)


# -----------------------------
# UI サイドバー
# -----------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### ⚙️ 設定")
    st.markdown("*Cài đặt*")
    st.markdown("#### 🎯 モード選択")
    mode = st.radio("Mode Selection", ["🗣️ 会話モード", "📝 テキスト翻訳", "🎤 音声入力", "📷 画像/PDF翻訳"], index=0, label_visibility="collapsed") or "🗣️ 会話モード"
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("#### 🌐 翻訳設定")
    st.markdown("*Cấu hình dịch*")
    
    # language row: [src] [⇄] [dst]
    col1, colS, col2 = st.columns([1, 0.3, 1])
    with col1:
        src_lang = st.selectbox("Source Language", ["🇻🇳 ベトナム語", "🇯🇵 日本語", "🇺🇸 英語", "🇧🇩 ベンガル語", "🇮🇩 インドネシア語"], 
                               index=["vi", "ja", "en", "bn", "id"].index(st.session_state.src), 
                               key="src_display", label_visibility="collapsed")
        # Update session state based on selection
        lang_map = {"🇻🇳 ベトナム語": "vi", "🇯🇵 日本語": "ja", "🇺🇸 英語": "en", "🇧🇩 ベンガル語": "bn", "🇮🇩 インドネシア語": "id"}
        if src_lang:
            st.session_state.src = lang_map[src_lang]
    
    with colS:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⇄", help="言語を入れ替え", use_container_width=True):
            swap_langs()
            st.rerun()
    
    with col2:
        dst_lang = st.selectbox("Target Language", ["🇯🇵 日本語", "🇻🇳 ベトナム語", "🇺🇸 英語", "🇧🇩 ベンガル語", "🇮🇩 インドネシア語"], 
                               index=["ja", "vi", "en", "bn", "id"].index(st.session_state.dst), 
                               key="dst_display", label_visibility="collapsed")
        if dst_lang:
            st.session_state.dst = lang_map[dst_lang]
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("#### 🎵 音声設定")
    st.markdown("*Cài đặt giọng nói*")
    
    voice_options = {"🤖 Alloy": "alloy", "🎭 Echo": "echo", "🎪 Fable": "fable", "🧙 Onyx": "onyx", "🎀 Nova": "nova", "🎵 Shimmer": "shimmer"}
    voice_choice = st.selectbox("音声タイプ", list(voice_options.keys()), index=0)
    tts_voice = voice_options[voice_choice] if voice_choice else "alloy"
    
    # Always use WAV format for better mobile compatibility
    st.info("🔊 音声形式: WAV (モバイル互換性のため)")
    audio_format = "wav"
    st.markdown('</div>', unsafe_allow_html=True)

# read current choices from session
src_choice = st.session_state.src
dst_choice = st.session_state.dst

if mode.startswith("📝"):
    st.markdown('<div class="mode-section">', unsafe_allow_html=True)
    st.markdown("## 📝 テキスト翻訳")
    st.markdown("*Dịch văn bản với phân tích AI*")
    
    # Dynamic example based on language settings
    if dst_choice == "ja":
        example = "Xin chào, rất vui được gặp bạn!" if src_choice == "vi" else ("Hello, nice to meet you!" if src_choice == "en" else "こんにちは、お会いできて嬉しいです！")
    elif dst_choice == "vi":
        example = "今日はとても暑いですね。" if src_choice == "ja" else ("The weather is very hot today." if src_choice == "en" else "Hôm nay thời tiết rất nóng.")
    else:  # dst_choice == "en"
        example = "今日はとても暑いですね。" if src_choice == "ja" else ("Xin chào, rất vui được gặp bạn!" if src_choice == "vi" else "Hello, how are you today?")
    
    text_in = st.text_area("✍️ テキストを入力してください", example, height=120, 
                          help="翻訳したいテキストを入力してください")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        translate_btn = st.button("🚀 AI翻訳を開始", type="primary", use_container_width=True)
    
    if translate_btn and text_in.strip():
        # AI Analysis Loading
        analysis_placeholder = st.empty()
        with analysis_placeholder:
            show_loading_animation("🧠 AI分析中", "文脈、丁寧度、調子を分析しています...")
        
        # First, detect context and formality
        detected_input = detect_lang_simple(text_in)
        
        # Vice versa translation logic - same as other modes
        if detected_input == src_choice:
            target_lang = dst_choice
        elif detected_input == dst_choice:
            target_lang = src_choice
        else:
            target_lang = dst_choice  # Default to destination language
        
        context_info = detect_formality_and_context(text_in, detected_input)
        analysis_placeholder.empty()
            
        # Show AI analysis in a more attractive format
        with st.expander("🔍 AI分析結果", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("検出言語", 
                        f"{detected_input.upper()}", 
                        delta="🔍")
            with col2:
                formality_emoji = {"casual": "😊", "neutral": "😐", "formal": "🎩", "very_formal": "👔"}
                formality_jp = {"casual": "カジュアル", "neutral": "普通", "formal": "丁寧", "very_formal": "非常に丁寧"}
                current_formality = context_info.get("formality", "neutral")
                st.metric("丁寧度", 
                        f"{formality_jp.get(current_formality, '普通')}", 
                        delta=f"{formality_emoji.get(current_formality, '😐')}")
            with col3:
                context_emoji = {"personal": "👥", "business": "💼", "academic": "🎓", "technical": "⚙️", "creative": "🎨", "medical": "🏥", "legal": "⚖️"}
                context_jp = {"personal": "個人的", "business": "ビジネス", "academic": "学術的", "technical": "技術的", "creative": "創作的", "medical": "医療", "legal": "法的"}
                current_context = context_info.get("context", "personal")
                st.metric("文脈", 
                        f"{context_jp.get(current_context, '個人的')}",
                        delta=f"{context_emoji.get(current_context, '👥')}")
            with col4:
                tone_emoji = {"friendly": "😊", "professional": "💼", "serious": "😐", "playful": "😄", "urgent": "⚡", "polite": "🙏"}
                tone_jp = {"friendly": "親しみやすい", "professional": "プロ的", "serious": "真面目", "playful": "遊び心", "urgent": "緊急", "polite": "礼儀正しい"}
                current_tone = context_info.get("tone", "friendly")
                st.metric("調子", 
                        f"{tone_jp.get(current_tone, '親しみやすい')}",
                        delta=f"{tone_emoji.get(current_tone, '😊')}")
        
        # Translation Loading
        translation_placeholder = st.empty()
        with translation_placeholder:
            show_loading_animation("✨ 高度AI翻訳中", "文脈を考慮した自然な翻訳を生成しています...")
        
        out = translate_text(text_in, detected_input, target_lang)
        translation_placeholder.empty()
        
        st.success(f"🎉 翻訳完了: {detected_input.upper()} → {target_lang.upper()}")
        
        # Display translation result in attractive format
        st.markdown("### 🎯 翻訳結果")
        out_safe = out.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#x27;')
        st.markdown(f"""
        <div class="translation-box">
            <div style="font-size: 1.4rem; line-height: 1.6;">
                <span style="background: rgba(255,255,255,0.2); padding: 0.2rem 0.6rem; 
                             border-radius: 15px; font-size: 0.9rem; margin-right: 1rem;">{target_lang.upper()}</span>
                {out_safe}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Audio output
        audio_bytes, mime = speak(out, voice=tts_voice, fmt=audio_format)
        if audio_bytes:
            play_audio_mobile_friendly(audio_bytes, mime)
    
    st.markdown('</div>', unsafe_allow_html=True)

elif mode.startswith("📷"):
    st.markdown('<div class="mode-section">', unsafe_allow_html=True)
    st.markdown("## 📷 画像/PDF翻訳")
    st.markdown("*Ảnh chụp / PDF → Dịch văn bản*")
    st.markdown("撮影した写真、アップロードした画像、またはPDFから文字を抽出して翻訳します。")

    col1, col2 = st.columns([1, 1])
    with col1:
        camera_image = st.camera_input("📸 カメラで撮影")
    with col2:
        upload_file = st.file_uploader(
            "📎 画像またはPDFをアップロード",
            type=["png", "jpg", "jpeg", "pdf"],
            help="画像(JPG/PNG)またはPDFを選択してください。",
        )

    translate_image_btn = st.button("🔍 OCRして翻訳", type="primary", use_container_width=True)

    if translate_image_btn:
        selected_file = camera_image if camera_image is not None else upload_file
        if selected_file is None:
            st.warning("⚠️ 画像またはPDFを選択してください。")
        else:
            file_bytes = selected_file.getvalue()
            file_type = selected_file.type or ""

            if file_type == "application/pdf" or selected_file.name.lower().endswith(".pdf"):
                extraction_placeholder = st.empty()
                with extraction_placeholder:
                    show_loading_animation("📄 PDF解析中", "PDFからテキストを抽出しています...")
                extracted_text = extract_text_from_pdf(file_bytes)
                extraction_placeholder.empty()
                if not extracted_text.strip():
                    st.warning("⚠️ PDFからテキストを抽出できませんでした。画像PDFの場合は画像としてアップロードしてください。")
                    extracted_text = ""
            else:
                extraction_placeholder = st.empty()
                with extraction_placeholder:
                    show_loading_animation("🖼️ 画像OCR中", "画像から文字を読み取っています...")
                extracted_text = extract_text_from_image(file_bytes, file_type or "image/png")
                extraction_placeholder.empty()

            if extracted_text.strip():
                st.markdown("### 📝 抽出されたテキスト")
                extracted_safe = extracted_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#x27;')
                st.markdown(f"""
                <div class="transcript-box">
                    <div style="font-size: 1.2rem; line-height: 1.5; color: #000;">
                        {extracted_safe}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                detected = detect_lang_simple(extracted_text)
                if detected == src_choice:
                    target = dst_choice
                elif detected == dst_choice:
                    target = src_choice
                else:
                    target = dst_choice

                translation_placeholder = st.empty()
                with translation_placeholder:
                    show_loading_animation("✨ 翻訳中", "抽出テキストを翻訳しています...")

                out = translate_text(extracted_text, detected, target)
                translation_placeholder.empty()

                st.markdown("### 🎯 翻訳結果")
                out_safe = out.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#x27;')
                st.markdown(f"""
                <div class="translation-box">
                    <div style="font-size: 1.4rem; line-height: 1.6;">
                        <span style="background: rgba(255,255,255,0.2); padding: 0.2rem 0.6rem;
                                     border-radius: 15px; font-size: 0.9rem; margin-right: 1rem;">{target.upper()}</span>
                        {out_safe}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                audio_bytes, mime = speak(out, voice=tts_voice, fmt=audio_format)
                if audio_bytes:
                    play_audio_mobile_friendly(audio_bytes, mime)

    st.markdown('</div>', unsafe_allow_html=True)

# Add other modes here (voice input, conversation) - complete implementation
elif mode.startswith("🎤"):
    st.markdown('<div class="mode-section">', unsafe_allow_html=True)
    st.markdown("## 🎤 音声入力翻訳")
    st.markdown("*Dịch đầu vào giọng nói với phân tích AI*")
    
    # Centered mic button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div style='text-align: center; padding: 2rem 0;'>", unsafe_allow_html=True)
        wav_bytes = audio_recorder(
            text="🎤 録音", 
            recording_color="#e53935", 
            neutral_color="#667eea", 
            icon_size="4x",
            pause_threshold=2.0,
            sample_rate=41_000
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #666; font-style: italic;'>マイクボタンを押して話してください</p>", unsafe_allow_html=True)
    
    if wav_bytes:
        # Validate audio data before processing
        if len(wav_bytes) < 1000:  # Too small to be valid audio
            st.warning("⚠️ 録音された音声が短すぎます。もう一度お試しください。")
        else:
            # Speech Recognition Loading
            recognition_placeholder = st.empty()
            with recognition_placeholder:
                show_loading_animation("🎧 音声認識中", "音声をテキストに変換しています...")
            
            transcript = transcribe_bytes(wav_bytes, "auto")
            recognition_placeholder.empty()
            
            if not transcript.strip():
                st.warning("⚠️ 音声を認識できませんでした。もう一度録音してください。")
            else:
                detected = detect_lang_simple(transcript)
                
                # Show transcript in attractive format
                st.markdown("### 📝 認識されたテキスト")
                transcript_safe = transcript.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#x27;')
                st.markdown(f"""
                <div class="transcript-box">
                    <div style="font-size: 1.3rem; line-height: 1.5; color: #000;">
                        <span style="background: #667eea; color: white; padding: 0.2rem 0.6rem; 
                                     border-radius: 15px; font-size: 0.9rem; margin-right: 1rem;">{detected.upper()}</span>
                        {transcript_safe}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Vice versa translation logic
                if detected == src_choice:
                    target = dst_choice
                elif detected == dst_choice:
                    target = src_choice
                else:
                    target = dst_choice

                # AI Context Analysis
                analysis_placeholder2 = st.empty()
                with analysis_placeholder2:
                    show_loading_animation("🧠 音声分析中", "話し方の調子と文脈を分析しています...")
                
                context_info = detect_formality_and_context(transcript, detected)
                analysis_placeholder2.empty()
                    
                with st.expander("🔍 AI分析結果", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        formality_emoji = {"casual": "😊", "neutral": "😐", "formal": "🎩", "very_formal": "👔"}
                        formality_jp = {"casual": "カジュアル", "neutral": "普通", "formal": "丁寧", "very_formal": "非常に丁寧"}
                        current_formality = context_info.get("formality", "neutral")
                        st.metric("丁寧度", 
                                f"{formality_jp.get(current_formality, '普通')}", 
                                delta=f"{formality_emoji.get(current_formality, '😐')}")
                    with col2:
                        context_emoji = {"personal": "👥", "business": "💼", "academic": "🎓", "technical": "⚙️", "creative": "🎨", "medical": "🏥", "legal": "⚖️"}
                        context_jp = {"personal": "個人的", "business": "ビジネス", "academic": "学術的", "technical": "技術的", "creative": "創作的", "medical": "医療", "legal": "法的"}
                        current_context = context_info.get("context", "personal")
                        st.metric("文脈", 
                                f"{context_jp.get(current_context, '個人的')}",
                                delta=f"{context_emoji.get(current_context, '👥')}")
                    with col3:
                        tone_emoji = {"friendly": "😊", "professional": "💼", "serious": "😐", "playful": "😄", "urgent": "⚡", "polite": "🙏"}
                        tone_jp = {"friendly": "親しみやすい", "professional": "プロ的", "serious": "真面目", "playful": "遊び心", "urgent": "緊急", "polite": "礼儀正しい"}
                        current_tone = context_info.get("tone", "friendly")
                        st.metric("調子", 
                                f"{tone_jp.get(current_tone, '親しみやすい')}",
                                delta=f"{tone_emoji.get(current_tone, '😊')}")

                # Voice Translation Loading
                voice_translation_placeholder = st.empty()
                with voice_translation_placeholder:
                    show_loading_animation("🗣️ 音声翻訳中", "自然で流暢な翻訳を生成しています...")
                
                out = translate_text(transcript, detected, target)
                voice_translation_placeholder.empty()
                
                # Display translation result
                st.markdown("### 🎯 翻訳結果")
                out_safe = out.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#x27;')
                st.markdown(f"""
                <div class="translation-box">
                    <div style="font-size: 1.4rem; line-height: 1.6;">
                        <span style="background: rgba(255,255,255,0.2); padding: 0.2rem 0.6rem; 
                                     border-radius: 15px; font-size: 0.9rem; margin-right: 1rem;">{target.upper()}</span>
                        {out_safe}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                audio_bytes, mime = speak(out, voice=tts_voice, fmt=audio_format)
                if audio_bytes:
                    play_audio_mobile_friendly(audio_bytes, mime)
    
    st.markdown('</div>', unsafe_allow_html=True)

elif mode.startswith("🗣️"):
    st.markdown('<div class="mode-section">', unsafe_allow_html=True)
    st.markdown("## 🗣️ リアルタイム会話翻訳")
    st.markdown("*Dịch hội thoại thời gian thực*")
    
    # Language selection directly in conversation mode
    st.markdown("#### 🌐 翻訳言語設定")
    col1, col_swap, col2 = st.columns([2, 0.8, 2])
    
    with col1:
        src_conv = st.selectbox(
            "入力言語 / Ngôn ngữ đầu vào:",
            ["🇻🇳 ベトナム語", "🇯🇵 日本語", "🇺🇸 英語", "🇧🇩 ベンガル語", "🇮🇩 インドネシア語"], 
            index=1,  # Default to Japanese
            key="conv_src"
        )
        # Update session state
        lang_map = {"🇻🇳 ベトナム語": "vi", "🇯🇵 日本語": "ja", "🇺🇸 英語": "en", "🇧🇩 ベンガル語": "bn", "🇮🇩 インドネシア語": "id"}
        if src_conv:
            st.session_state.src = lang_map[src_conv]
    
    with col_swap:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 言語交換", help="入力と出力言語を入れ替え", use_container_width=True, key="conv_swap"):
            swap_langs()
            st.rerun()
    
    with col2:
        dst_conv = st.selectbox(
            "出力言語 / Ngôn ngữ đầu ra:",
            ["🇯🇵 日本語", "🇻🇳 ベトナム語", "🇺🇸 英語", "🇧🇩 ベンガル語", "🇮🇩 インドネシア語"], 
            index=1,  # Default to Vietnamese
            key="conv_dst"
        )
        if dst_conv:
            st.session_state.dst = lang_map[dst_conv]
    
    # Show current language settings with badges
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.markdown(f'<span class="language-badge">入力: {st.session_state.src.upper()}</span>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div style="text-align: center; font-size: 1.5rem;">⇄</div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<span class="language-badge">出力: {st.session_state.dst.upper()}</span>', unsafe_allow_html=True)
    
    if "chat" not in st.session_state:
        st.session_state.chat = []
    
    # Centered large mic button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div style='text-align: center; padding: 2rem 0;'>", unsafe_allow_html=True)
        wav_bytes = audio_recorder(
            text="🎤 話す", 
            recording_color="#e53935", 
            neutral_color="#667eea", 
            icon_size="4x",
            pause_threshold=2.0,
            sample_rate=41_000,
            key="conversation_recorder"
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #666; font-style: italic;'>マイクボタンを押して話してください</p>", unsafe_allow_html=True)
    
    if wav_bytes:
        # Validate audio data before processing
        if len(wav_bytes) < 1000:
            st.warning("⚠️ 録音された音声が短すぎます。もう一度お試しください。")
        else:
            # Speech Recognition Loading
            recognition_placeholder = st.empty()
            with recognition_placeholder:
                show_loading_animation("🎧 音声認識中", "音声をテキストに変換しています...")
            
            transcript = transcribe_bytes(wav_bytes, "auto")
            recognition_placeholder.empty()
            
            if not transcript.strip():
                st.warning("⚠️ 音声を認識できませんでした。もう一度録音してください。")
            else:
                detected = detect_lang_simple(transcript)
                
                # Vice versa translation based on translation settings
                if detected == src_choice:
                    target = dst_choice
                elif detected == dst_choice:
                    target = src_choice
                else:
                    target = dst_choice
                
                # Real-time Translation Loading
                translation_placeholder = st.empty()
                with translation_placeholder:
                    show_loading_animation("🗣️ リアルタイム翻訳中", "会話を自然に翻訳しています...")
                    
                translation = translate_text(transcript, detected, target)
                translation_placeholder.empty()
                
                # Show success notification
                st.success(f"🎉 翻訳完了: {detected.upper()} → {target.upper()}")
                
                # Show current recognition and translation
                with st.expander("📝 現在の音声認識・翻訳結果", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**🎤 認識されたテキスト:**")
                        transcript_safe = transcript.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#x27;')
                        st.markdown(f"""
                        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; 
                                    border-left: 3px solid #667eea; margin: 0.5rem 0; color: #000;">
                            <span style="background: #667eea; color: white; padding: 0.2rem 0.6rem; 
                                         border-radius: 15px; font-size: 0.8rem; margin-right: 0.5rem;">{detected.upper()}</span>
                            {transcript_safe}
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown("**✨ 翻訳結果:**")
                        translation_safe = translation.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#x27;')
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    color: white; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
                            <span style="background: rgba(255,255,255,0.2); padding: 0.2rem 0.6rem; 
                                         border-radius: 15px; font-size: 0.8rem; margin-right: 0.5rem;">{target.upper()}</span>
                            {translation_safe}
                        </div>
                        """, unsafe_allow_html=True)
                
                st.session_state.chat.append({
                    "speaker": "A" if (len(st.session_state.chat) % 2 == 0) else "B",
                    "transcript": transcript,
                    "translation": translation,
                    "src": detected,
                    "dst": target,
                })
                
                audio_bytes, mime = speak(translation, voice=tts_voice, fmt=audio_format)
                if audio_bytes:
                    play_audio_mobile_friendly(audio_bytes, mime)

    # Conversation history
    if st.session_state.chat:
        st.markdown("### 💬 会話履歴")
        for i, msg in enumerate(reversed(st.session_state.chat)):
            role = msg["speaker"]
            turn_num = len(st.session_state.chat) - i
            
            with st.container():
                # Header with speaker info
                col1, col2 = st.columns([2, 1])
                with col1:
                    if role == "A":
                        st.markdown(f"**👤 話者 A · ターン {turn_num}**")
                    else:
                        st.markdown(f"**👤 話者 B · ターン {turn_num}**")
                with col2:
                    st.markdown(f"*{msg['src'].upper()} → {msg['dst'].upper()}*")
                
                # Original text
                st.markdown("**原文:**")
                transcript_safe = msg['transcript'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#x27;')
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; 
                            border-left: 3px solid #dee2e6; font-size: 1.1rem; margin: 0.5rem 0;
                            color: #333;">
                    {transcript_safe}
                </div>
                """, unsafe_allow_html=True)
                
                # Translation
                st.markdown("**翻訳:**")
                translation_safe = msg['translation'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#x27;')
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;
                            padding: 1.5rem; border-radius: 15px; font-size: 1.2rem; 
                            font-weight: 500; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15); margin: 0.5rem 0;">
                    {translation_safe}
                </div>
                """, unsafe_allow_html=True)
                
                st.divider()
    
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem 0; background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%); 
            border-radius: 10px; margin-top: 3rem;">
    <h4 style="color: #667eea; margin-bottom: 1rem;">🤖 AI-Powered Translation Assistant</h4>
    <p style="color: #666; margin: 0.5rem 0;">
        <strong>✨ 機能:</strong> 文脈認識翻訳 • 音声認識 • 画像/PDF翻訳 • リアルタイム会話
    </p>
    <p style="color: #666; margin: 0.5rem 0;">
        <strong>🔧 技術:</strong> Streamlit + OpenAI GPT-4o-mini • Python
    </p>
    <p style="color: #888; font-size: 0.9rem; margin-top: 1rem;">
        ベトナム語 ⇄ 日本語 ⇄ 英語 ⇄ ベンガル語 ⇄ インドネシア語翻訳システム
    </p>
</div>
""", unsafe_allow_html=True)
