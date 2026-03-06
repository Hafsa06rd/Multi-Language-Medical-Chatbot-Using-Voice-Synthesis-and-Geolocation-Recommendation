# app.py - Enhanced with Google Gemini API Integration
import streamlit as st
from ChatBot import MedicalChatbot
import speech_recognition as sr
import os
import soundfile as sf
import io
import tempfile
import pandas as pd
import pyaudio
import wave
import time


class StreamlitMedicalInterface:
    def __init__(self):
        self.chatbot = MedicalChatbot()
        self.recognizer = sr.Recognizer()
        self.load_dataset_info()

    def load_dataset_info(self):
        """Load and prepare dataset statistics for display"""
        try:
            self.df = pd.read_csv('Symptom_dataset.csv')
            self.dataset_stats = {
                'total_cases': len(self.df),
                'unique_diseases': len(self.df['Disease'].unique()),
                'age_range': f"{self.df['Age'].min()} - {self.df['Age'].max()}",
                'gender_dist': self.df['Gender'].value_counts().to_dict()
            }
        except Exception as e:
            print(f"Error loading dataset statistics: {e}")
            self.dataset_stats = None

    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
            # Add enhanced initial greeting message hi
            greeting = {
                "role": "assistant",
                "content": "🩺 **Welcome to your Enhanced Multilingual Medical AI Assistant!**\n\nI'm here to provide you with professional-grade medical insights. I can help you understand your symptoms, suggest appropriate specialists, and provide evidence-based recommendations.\n\n**How can I help you today?**\n\n*Feel free to describe your symptoms in detail, or use the voice input for hands-free interaction.*"
            }
            st.session_state.messages.append(greeting)
        if 'current_language' not in st.session_state:
            st.session_state.current_language = 'en'
        if 'show_dataset_info' not in st.session_state:
            st.session_state.show_dataset_info = False
        if 'api_configured' not in st.session_state:
            st.session_state.api_configured = self.chatbot.gemini.is_configured()
        if 'show_help' not in st.session_state:
            st.session_state.show_help = False

    def display_header(self):
        """Display the enhanced application header"""
        # Custom CSS for better styling
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 1rem;
        }
        .status-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 15px;
            font-size: 0.8rem;
            font-weight: bold;
            margin: 0.25rem;
        }
        .status-success {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .status-warning {
            background-color: #fff3cd;
            color: #856404;
            border: 1px solid #ffeaa7;
        }
        .feature-card {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            margin: 0.5rem 0;
        }
        </style>
        """, unsafe_allow_html=True)

        # Main header
        st.markdown("""
        <div class="main-header">
            <h1>🩺 Enhanced Multilingual AI Medical Chatbot With Voice Synthesis</h1>
            <p>Professional Multilingual Medical Analysis with Voice Support</p>
        </div>
        """, unsafe_allow_html=True)

        # API Status Section
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            if st.session_state.api_configured:
                st.markdown("""
                <div class="status-badge status-success">
                    ✅  Active - Professional Analysis Enabled
                </div>
                """, unsafe_allow_html=True)
                st.success(
                    "🚀 **Enhanced Features Active**: Advanced symptom analysis, professional recommendations, urgency assessment, and specialist referrals !")
            else:
                st.markdown("""
                <div class="status-badge status-warning">
                    ⚠️ Using Local Analysis Only - Not Configured
                </div>
                """, unsafe_allow_html=True)

                with st.expander("🔧 **Enable Professional AI Analysis** (Free & Easy!)", expanded=False):
                    st.markdown("""
                    **🎯 Benefits of AI Enhancement:**
                    - ⚡ **Smart urgency assessment** (Emergency/Urgent/Routine)
                    - 👨‍⚕️ **Automatic specialist referrals**
                    - 📋 **Evidence-based recommendations**
                    - 🌍 **Works in all supported languages**
                    """)

        with col2:
            if st.button("📊 Dataset Info", help="View medical dataset statistics"):
                st.session_state.show_dataset_info = not st.session_state.show_dataset_info

        with col3:
            if st.button("💡 Usage Tips", help="Get tips for better diagnosis"):
                st.session_state.show_help = not st.session_state.show_help

        # Dataset Information Section
        if st.session_state.show_dataset_info and self.dataset_stats:
            st.markdown("### 📈 Medical Dataset Statistics")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📋 Total Cases", self.dataset_stats['total_cases'])
            with col2:
                st.metric("🏥 Unique Diseases", self.dataset_stats['unique_diseases'])
            with col3:
                st.metric("👥 Age Range", self.dataset_stats['age_range'])
            with col4:
                st.metric("⚖️ Gender Split", f"{len(self.dataset_stats['gender_dist'])} types")

            # Gender distribution chart
            if self.dataset_stats['gender_dist']:
                st.markdown("**Gender Distribution:**")
                gender_data = self.dataset_stats['gender_dist']
                col1, col2 = st.columns(2)
                for i, (gender, count) in enumerate(gender_data.items()):
                    with col1 if i % 2 == 0 else col2:
                        st.write(f"• **{gender}**: {count} cases")

        # Usage Tips Section
        if st.session_state.show_help:
            st.markdown("### 💡 Tips for Accurate Medical Analysis")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                <div class="feature-card">
                <h4>🎯 Be Specific & Detailed</h4>
                <ul>
                <li><strong>Include severity</strong>: "severe headache" vs "headache"</li>
                <li><strong>Mention duration</strong>: "3 days" or "since yesterday"</li>
                <li><strong>Add context</strong>: "after eating" or "when standing"</li>
                <li><strong>Describe quality</strong>: "throbbing", "sharp", "dull"</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div class="feature-card">
                <h4>📝 Example Good Inputs</h4>
                <ul>
                <li>"I'm 25 with severe throbbing headaches for 2 days, nausea, and light sensitivity"</li>
                <li>"Sharp chest pain when breathing deeply, started this morning"</li>
                <li>"Persistent dry cough for a week, slight fever, very tired"</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown("""
                <div class="feature-card">
                <h4>🌍 Multilingual Support</h4>
                <ul>
                <li>🇺🇸 English</li>
                <li>🇫🇷 French</li>
                <li>🇪🇸 Spanish</li>
                <li>🇩🇪 German</li>
                <li>🇮🇹 Italian</li>
                <li>🇵🇹 Portuguese</li>
                <li>🇸🇦 Arabic</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div class="feature-card">
                <h4>🎤 Voice Input Tips</h4>
                <ul>
                <li><strong>Speak clearly</strong> and at normal pace</li>
                <li><strong>Minimize background noise</strong></li>
                <li><strong>Use your selected language</strong></li>
                <li><strong>Speak for 3-5 seconds</strong> when prompted</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

        # Language Selector
        st.markdown("### 🌍 Language Preferences")
        col1, col2 = st.columns([3, 1])

        with col1:
            selected_language = st.selectbox(
                "Select your preferred language for interaction:",
                options=list(self.chatbot.supported_languages.keys()),
                format_func=lambda
                    x: f"{self.chatbot.supported_languages[x].title()} {'🇺🇸' if x == 'en' else '🇫🇷' if x == 'fr' else '🇪🇸' if x == 'es' else '🇩🇪' if x == 'de' else '🇮🇹' if x == 'it' else '🇵🇹' if x == 'pt' else '🇸🇦'}",
                index=list(self.chatbot.supported_languages.keys()).index(st.session_state.current_language),
                help="All AI features work in your selected language"
            )

        if selected_language != st.session_state.current_language:
            st.session_state.current_language = selected_language
            self.translate_chat_history(selected_language)
            st.rerun()

        # Welcome message in selected language
        welcome_message = self.chatbot.translate_text(
            "I'm ready to help you analyze your symptoms with advanced AI. Describe how you're feeling, and I'll provide professional medical insights and recommendations in your language.",
            selected_language
        )

        st.info(f"🤖 **AI Assistant**: {welcome_message}")

    def translate_chat_history(self, target_lang: str):
        """Translate entire chat history to new language"""
        with st.spinner("🔄 Translating conversation history..."):
            for message in st.session_state.messages:
                if message["role"] == "assistant":
                    # Only translate if it's not already in target language
                    message["content"] = self.chatbot.translate_text(message["content"], target_lang)

    def record_audio(self):
        """Enhanced audio recording with better UX"""
        try:
            CHUNK = 1024
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            RATE = 44100
            RECORD_SECONDS = 5

            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
                temp_filename = temp_audio.name

            # Initialize PyAudio
            p = pyaudio.PyAudio()

            # Open stream
            stream = p.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)

            # Enhanced recording UI
            st.markdown("### 🎤 **Voice Recording Active**")

            # Countdown and progress
            countdown_placeholder = st.empty()
            progress_bar = st.progress(0)
            status_placeholder = st.empty()

            frames = []

            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)

                # Update progress and countdown
                progress = (i + 1) / (RATE / CHUNK * RECORD_SECONDS)
                remaining_time = RECORD_SECONDS - (progress * RECORD_SECONDS)

                progress_bar.progress(progress)
                countdown_placeholder.markdown(f"**🎙️ Recording... {remaining_time:.1f}s remaining**")
                status_placeholder.markdown("*Speak clearly about your symptoms...*")

            # Processing phase
            countdown_placeholder.markdown("**🔄 Processing your voice...**")
            status_placeholder.markdown("*Converting speech to text...*")

            # Stop recording
            stream.stop_stream()
            stream.close()
            p.terminate()

            # Save the audio file
            wf = wave.open(temp_filename, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            # Convert speech to text
            with sr.AudioFile(temp_filename) as source:
                audio = self.recognizer.record(source)
                text = self.recognizer.recognize_google(
                    audio,
                    language=st.session_state.current_language
                )

            # Clean up
            os.unlink(temp_filename)

            # Clear the recording UI
            countdown_placeholder.empty()
            progress_bar.empty()
            status_placeholder.empty()

            return text

        except sr.UnknownValueError:
            st.error("🎤 **Could not understand audio**. Please try again and speak more clearly.")
            return None
        except sr.RequestError as e:
            st.error(f"🌐 **Speech recognition service error**: {e}")
            return None
        except Exception as e:
            st.error(f"❌ **Recording error**: {str(e)}")
            return None

    def process_input(self, user_input: str):
        """Enhanced input processing with AI analysis indicators"""
        try:
            # Show AI analysis progress
            if self.chatbot.gemini.is_configured():
                with st.spinner('🧠 **AI Analysis in Progress**'):
                    st.markdown("*Analyzing your symptoms...*")
                    # Process input with AI enhancement
                    response, detected_lang, audio_path = self.chatbot.process_input(user_input)
            else:
                with st.spinner('🔍 **Analyzing symptoms locally**'):
                    st.markdown("*Using local medical knowledge base...*")
                    response, detected_lang, audio_path = self.chatbot.process_input(user_input)

            # Add enhanced dataset insights if available
            if self.df is not None:
                try:
                    # Extract potential disease mentions
                    response_lower = response.lower()
                    potential_diseases = []

                    for disease in self.df['Disease'].unique():
                        if disease.lower() in response_lower:
                            potential_diseases.append(disease)

                    if potential_diseases:
                        # Get statistics for the first matching disease
                        disease = potential_diseases[0]
                        disease_data = self.df[self.df['Disease'] == disease]

                        if not disease_data.empty:
                            avg_age = disease_data['Age'].mean()
                            gender_mode = disease_data['Gender'].mode().iloc[0] if not disease_data[
                                'Gender'].mode().empty else 'Not specified'
                            case_count = len(disease_data)

                            dataset_insights = f"""

📊 **Dataset Insights for {disease}:**
• **Average patient age**: {avg_age:.1f} years
• **Most common in**: {gender_mode}
• **Cases in our database**: {case_count}
• **Confidence boost**: Analysis backed by real patient data
                            """
                            response += dataset_insights

                except Exception as e:
                    print(f"Error adding dataset insights: {e}")

            # Create response message
            message = {
                "role": "assistant",
                "content": response,
                "timestamp": time.time()
            }

            # Add audio if available
            if audio_path and os.path.exists(audio_path):
                try:
                    with open(audio_path, 'rb') as audio_file:
                        message["audio_data"] = audio_file.read()
                    os.remove(audio_path)
                except Exception as e:
                    print(f"Error handling audio: {e}")

            st.session_state.messages.append(message)

        except Exception as e:
            st.error(f"❌ **Analysis Error**: {e}")
            error_message = {
                "role": "assistant",
                "content": f"I apologize, but I encountered an error while analyzing your symptoms. Please try rephrasing your symptoms or try again.\n\n**Error details**: {str(e)}",
                "timestamp": time.time()
            }
            st.session_state.messages.append(error_message)

    def display_chat_interface(self):
        """Enhanced chat interface with professional medical styling"""
        st.markdown("### 💬 Medical Consultation Chat")

        # Display chat history with enhanced formatting
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    # Enhanced formatting for AI responses
                    st.markdown(message["content"])

                    # Show AI indicator for enhanced responses
                    if self.chatbot.gemini.is_configured() and i > 0:  # Skip initial greeting
                        st.caption("🤖 *Medical Analysis *")
                else:
                    # User message formatting
                    st.markdown(f"**You said:** {message['content']}")

                # Audio playback
                if "audio_data" in message:
                    try:
                        st.audio(message["audio_data"], format="audio/mp3")
                        st.caption("🔊 *Click to hear the audio response*")
                    except Exception as e:
                        print(f"Error playing audio: {e}")

        # Enhanced input controls
        st.markdown("---")
        st.markdown("### 🎯 **Input Options**")

        # Control buttons
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("🎤 **Voice Input**", help="Record your symptoms using voice", use_container_width=True):
                text = self.record_audio()
                if text:
                    st.success(f"✅ **Recorded**: {text}")
                    st.session_state.messages.append({
                        "role": "user",
                        "content": text,
                        "timestamp": time.time()
                    })
                    self.process_input(text)
                    st.rerun()

        with col2:
            if st.button("🗑️ **Clear Chat**", help="Start a new consultation", use_container_width=True):
                st.session_state.messages = []
                # Add enhanced greeting
                greeting = {
                    "role": "assistant",
                    "content": "🩺 **New Consultation Started**\n\nI'm ready to help analyze your symptoms. Please describe how you're feeling in detail.",
                    "timestamp": time.time()
                }
                st.session_state.messages.append(greeting)
                st.rerun()

        with col3:
            if st.button("📋 **Example Symptoms**", help="See example inputs", use_container_width=True):
                examples = [
                    "I have a severe headache for 2 days with nausea and light sensitivity",
                    "Sharp chest pain when breathing, started this morning",
                    "Persistent dry cough for a week, slight fever, very tired",
                    "Stomach pain after eating, bloated feeling, loose stools"
                ]
                example = st.selectbox("Choose an example:", examples, key="example_select")
                if st.button("Use This Example"):
                    st.session_state.messages.append({
                        "role": "user",
                        "content": example,
                        "timestamp": time.time()
                    })
                    self.process_input(example)
                    st.rerun()

        with col4:
            if st.button("📞 **Emergency Info**", help="Emergency contact information", use_container_width=True):
                st.error("""
                🚨 **EMERGENCY SITUATIONS**

                **Call immediately if you experience:**
                • Chest pain or pressure
                • Difficulty breathing
                • Severe allergic reactions
                • Loss of consciousness
                • Severe bleeding
                • Signs of stroke

                **Emergency Numbers:**
                • MR: 15
                • US: 911
                • EU: 112
                • UK: 999
                """)

        # Main text input with enhanced placeholder
        placeholder_examples = [
            "Describe your symptoms in detail (e.g., 'severe headache for 2 days with nausea and light sensitivity')",
            "Tell me about your symptoms, including severity and duration",
            "What symptoms are you experiencing? Be as specific as possible",
            "Describe how you're feeling, when it started, and how severe it is"
        ]

        current_placeholder = placeholder_examples[len(st.session_state.messages) % len(placeholder_examples)]

        if prompt := st.chat_input(current_placeholder):
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": prompt,
                "timestamp": time.time()
            })

            # Process and respond
            self.process_input(prompt)
            st.rerun()

    def display_footer(self):
        """Enhanced footer with comprehensive medical disclaimers"""
        st.markdown("---")

        # Medical disclaimer
        st.markdown("""
        <div style='background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 1rem; margin: 1rem 0;'>
            <h4 style='color: #856404; margin-top: 0;'>⚠️ IMPORTANT MEDICAL DISCLAIMER</h4>
            <p style='color: #856404; margin-bottom: 0;'>
                <strong>This AI chatbot is for educational and informational purposes only.</strong> It is not intended to be a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Emergency warning
        st.markdown("""
        <div style='background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 8px; padding: 1rem; margin: 1rem 0;'>
            <h4 style='color: #721c24; margin-top: 0;'>🚨 EMERGENCY SITUATIONS</h4>
            <p style='color: #721c24; margin-bottom: 0;'>
                <strong>In case of a medical emergency, immediately call your local emergency services or go to the nearest emergency room. Do not rely on this chatbot for emergency medical situations.</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # AI and privacy info
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **🤖 AI Technology:**
            - Local medical dataset integration
            - Voice support
            - Multilingual support
            - Privacy-focused design
            """)

        with col2:
            st.markdown("""
            **🔒 Privacy & Security:**
            - No conversation data stored
            - Anonymous symptom processing
            - Local audio processing
            - HIPAA-conscious design
            """)

        # Version and credits
        st.markdown("""
        <div style='text-align: center; color: #666; font-size: 0.9em; padding: 1rem;'>
            <strong>Enhanced Medical Chatbot v2.0</strong><br>
            Multilingual/Voice support | Built with Streamlit<br>
            <em>Professional medical insights at your fingertips</em>
        </div>
        """, unsafe_allow_html=True)

    def run(self):
        """Run the complete enhanced Streamlit interface"""
        # Configure page settings
        st.set_page_config(
            page_title="AI Medical Chatbot - Professional Diagnosis Assistant",
            page_icon="🩺",
            layout="wide",
            initial_sidebar_state="collapsed",
            menu_items={
                'Get help': 'https://ai.google.dev/',
                'Report a bug': None,
                'About': "Enhanced Multilingual Medical Chatbot with Voice Support for professional medical analysis."
            }
        )

        # Initialize everything
        self.initialize_session_state()

        # Display all sections
        self.display_header()
        self.display_chat_interface()
        self.display_footer()


if __name__ == "__main__":
    # Run the enhanced medical interface
    interface = StreamlitMedicalInterface()
    interface.run()