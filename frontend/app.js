// BioChat AI - Biomedical Literature Mining RAG Application
// Complete frontend implementation with simulated backend

// Add near top
const API_BASE = 'http://localhost:8000';

class BioChatApp {
    constructor() {
        this.currentUser = null;
        this.chatHistory = [];
        this.currentMessages = [];
        this.isTyping = false;
        this.sidebarOpen = false;
        
        this.init();
    }

    init() {
        this.showLoadingScreen();
        
        // Simulate initialization delay
        setTimeout(() => {
            this.hideLoadingScreen();
            this.checkAuth();
            this.setupEventListeners();
            this.loadChatHistory();
        }, 2000);
    }

    showLoadingScreen() {
        document.getElementById('loading-screen').classList.remove('hidden');
        document.getElementById('auth-page').classList.add('hidden');
        document.getElementById('main-app').classList.add('hidden');
    }

    hideLoadingScreen() {
        document.getElementById('loading-screen').classList.add('hidden');
    }

    checkAuth() {
        const token = localStorage.getItem('bioChat_token');
        const user = localStorage.getItem('bioChat_user');
        
        if (token && user) {
            this.currentUser = JSON.parse(user);
            this.showMainApp();
        } else {
            this.showAuthPage();
        }
    }

    showAuthPage() {
        document.getElementById('auth-page').classList.remove('hidden');
        document.getElementById('main-app').classList.add('hidden');
    }

    showMainApp() {
        document.getElementById('auth-page').classList.add('hidden');
        document.getElementById('main-app').classList.remove('hidden');
        this.updateUserProfile();
        this.renderChatHistory();
    }

    setupEventListeners() {
        // Auth form handling
        const authForm = document.getElementById('auth-form');
        const loginTab = document.getElementById('login-tab');
        const signupTab = document.getElementById('signup-tab');
        const useDemoBtn = document.getElementById('use-demo');

        authForm.addEventListener('submit', (e) => this.handleAuth(e));
        loginTab.addEventListener('click', () => this.switchAuthTab('login'));
        signupTab.addEventListener('click', () => this.switchAuthTab('signup'));
        useDemoBtn.addEventListener('click', () => this.useDemoCredentials());

        // Chat interface
        const chatInput = document.getElementById('chat-input');
        const sendButton = document.getElementById('send-button');
        const newChatBtn = document.getElementById('new-chat');
        const logoutBtn = document.getElementById('logout');
        const sidebarToggle = document.getElementById('toggle-sidebar');

        chatInput.addEventListener('input', () => this.handleInputChange());
        chatInput.addEventListener('keydown', (e) => this.handleKeyDown(e));
        sendButton.addEventListener('click', () => this.sendMessage());
        newChatBtn.addEventListener('click', () => this.startNewChat());
        logoutBtn.addEventListener('click', () => this.logout());
        sidebarToggle.addEventListener('click', () => this.toggleSidebar());

        // Query suggestions
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('query-suggestion')) {
                this.handleQuerySuggestion(e.target.dataset.query);
            }
            if (e.target.classList.contains('follow-up-question')) {
                this.handleFollowUpQuestion(e.target.textContent);
            }
        });

        // Auto-resize textarea
        chatInput.addEventListener('input', () => {
            chatInput.style.height = 'auto';
            chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
        });
    }

    switchAuthTab(mode) {
        const loginTab = document.getElementById('login-tab');
        const signupTab = document.getElementById('signup-tab');
        const confirmPasswordGroup = document.getElementById('confirm-password-group');
        const submitBtn = document.getElementById('auth-submit');

        if (mode === 'login') {
            loginTab.classList.add('active');
            signupTab.classList.remove('active');
            confirmPasswordGroup.classList.add('hidden');
            submitBtn.textContent = 'Login';
        } else {
            signupTab.classList.add('active');
            loginTab.classList.remove('active');
            confirmPasswordGroup.classList.remove('hidden');
            submitBtn.textContent = 'Sign Up';
        }
    }

    useDemoCredentials() {
        document.getElementById('email').value = 'user@example.com';
        document.getElementById('password').value = 'password123';
        
        // Auto-login with demo credentials
        setTimeout(() => {
            this.handleAuth(new Event('submit'), true);
        }, 500);
    }

    async handleAuth(e, skipValidation = false) {
        e.preventDefault();
        
        const email = document.getElementById('email').value;
        const password = document.getElementById('password').value;
        const isSignup = document.getElementById('signup-tab').classList.contains('active');
        
        if (!skipValidation) {
            if (isSignup) {
                const confirmPassword = document.getElementById('confirm-password').value;
                if (password !== confirmPassword) {
                    this.showAuthError('Passwords do not match');
                    return;
                }
            }
        }

        // Call API
        try {
            const response = await this.simulateAuthAPI(email, password, isSignup);
            
            if (response.success) {
                this.currentUser = response.user;
                localStorage.setItem('bioChat_token', response.token);
                localStorage.setItem('bioChat_user', JSON.stringify(response.user));
                this.showMainApp();
            } else {
                this.showAuthError(response.error);
            }
        } catch (error) {
            this.showAuthError('Authentication failed. Please try again.');
        }
    }

    // Replace simulateAuthAPI with live backend
    async simulateAuthAPI(email, password, isSignup) {
        const endpoint = isSignup ? '/api/auth/signup' : '/api/auth/login';
        const body = isSignup
            ? { email, password, confirm_password: password }
            : { email, password };

        const res = await fetch(`${API_BASE}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });

        const data = await res.json();
        if (res.ok && data.success) {
            return { success: true, token: data.token, user: data.user };
        } else {
            return { success: false, error: data.detail || 'Authentication failed' };
        }
    }

    showAuthError(message) {
        const errorDiv = document.getElementById('auth-error');
        errorDiv.textContent = message;
        errorDiv.classList.remove('hidden');
        
        setTimeout(() => {
            errorDiv.classList.add('hidden');
        }, 5000);
    }

    updateUserProfile() {
        if (this.currentUser) {
            document.getElementById('user-name').textContent = this.currentUser.name;
            document.getElementById('user-email').textContent = this.currentUser.email;
        }
    }

    handleInputChange() {
        const input = document.getElementById('chat-input');
        const sendBtn = document.getElementById('send-button');
        
        sendBtn.disabled = !input.value.trim();
    }

    handleKeyDown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            this.sendMessage();
        }
    }

    handleQuerySuggestion(query) {
        document.getElementById('chat-input').value = query;
        this.handleInputChange();
        this.sendMessage();
    }

    handleFollowUpQuestion(question) {
        document.getElementById('chat-input').value = question;
        this.handleInputChange();
        this.sendMessage();
    }

    async sendMessage() {
        const input = document.getElementById('chat-input');
        const message = input.value.trim();
        
        if (!message) return;

        // Clear input and disable send button
        input.value = '';
        input.style.height = 'auto';
        this.handleInputChange();

        // Add user message
        this.addMessage('user', message);
        
        // Show typing indicator
        this.showTypingIndicator();
        
        // Call API
        try {
            const response = await this.simulateQueryAPI(message);
            this.hideTypingIndicator();
            this.addMessage('ai', response.answer, response);
            
            // Update chat history
            this.updateChatHistory(message, response.answer);
            
        } catch (error) {
            this.hideTypingIndicator();
            this.addMessage('ai', 'I apologize, but I encountered an error processing your query. Please try again.', {
                confidence_score: 0,
                processing_time: 0,
                citations: [],
                follow_up_questions: []
            });
        }
    }

    addMessage(type, content, metadata = null) {
        const messagesContainer = document.getElementById('chat-messages');
        
        // Remove welcome message if it exists
        const welcomeMsg = messagesContainer.querySelector('.welcome-message');
        if (welcomeMsg) {
            welcomeMsg.remove();
        }

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        
        const bubbleDiv = document.createElement('div');
        bubbleDiv.className = 'message-bubble';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.innerHTML = this.formatMessageContent(content);
        
        bubbleDiv.appendChild(contentDiv);

        if (type === 'ai' && metadata) {
            // Add metadata
            if (metadata.confidence_score > 0 || metadata.processing_time > 0) {
                const metaDiv = document.createElement('div');
                metaDiv.className = 'message-meta';
                
                if (metadata.confidence_score > 0) {
                    const confidenceSpan = document.createElement('span');
                    confidenceSpan.className = 'confidence-score';
                    confidenceSpan.textContent = `Confidence: ${(metadata.confidence_score * 100).toFixed(0)}%`;
                    metaDiv.appendChild(confidenceSpan);
                }
                
                if (metadata.processing_time > 0) {
                    const timeSpan = document.createElement('span');
                    timeSpan.className = 'processing-time';
                    timeSpan.textContent = `${metadata.processing_time.toFixed(2)}s`;
                    metaDiv.appendChild(timeSpan);
                }
                
                bubbleDiv.appendChild(metaDiv);
            }

            // Add citations
            if (metadata.citations && metadata.citations.length > 0) {
                const citationsDiv = document.createElement('div');
                citationsDiv.className = 'citations';
                
                const titleDiv = document.createElement('div');
                titleDiv.className = 'citations-title';
                titleDiv.textContent = 'Sources';
                citationsDiv.appendChild(titleDiv);
                
                metadata.citations.forEach(citation => {
                    const citationLink = document.createElement('a');
                    citationLink.className = 'citation-item';
                    citationLink.href = citation.doi ? `https://doi.org/${citation.doi}` : `https://www.ncbi.nlm.nih.gov/pmc/articles/${citation.pmcid}/`;
                    citationLink.target = '_blank';
                    
                    const titleSpan = document.createElement('div');
                    titleSpan.className = 'citation-title';
                    titleSpan.textContent = citation.title;
                    
                    const metaSpan = document.createElement('div');
                    metaSpan.className = 'citation-meta';
                    metaSpan.textContent = `${citation.journal} (${citation.year}) • ${citation.pmcid || citation.doi}`;
                    
                    citationLink.appendChild(titleSpan);
                    citationLink.appendChild(metaSpan);
                    citationsDiv.appendChild(citationLink);
                });
                
                bubbleDiv.appendChild(citationsDiv);
            }

            // Add follow-up questions
            if (metadata.follow_up_questions && metadata.follow_up_questions.length > 0) {
                const followUpDiv = document.createElement('div');
                followUpDiv.className = 'follow-up-questions';
                
                const titleDiv = document.createElement('div');
                titleDiv.className = 'follow-up-title';
                titleDiv.textContent = 'Related Questions';
                followUpDiv.appendChild(titleDiv);
                
                metadata.follow_up_questions.forEach(question => {
                    const questionBtn = document.createElement('button');
                    questionBtn.className = 'follow-up-question';
                    questionBtn.textContent = question;
                    followUpDiv.appendChild(questionBtn);
                });
                
                bubbleDiv.appendChild(followUpDiv);
            }
        }

        messageDiv.appendChild(bubbleDiv);
        messagesContainer.appendChild(messageDiv);
        
        // Store message
        this.currentMessages.push({
            type,
            content,
            metadata,
            timestamp: new Date()
        });

        // Auto-scroll
        this.scrollToBottom();
    }

    formatMessageContent(content) {
        // Handle citation brackets [1], [2], etc.
        return content.replace(/\[(\d+)\]/g, '<sup style="color: #FFD700; font-weight: 600;">[$1]</sup>');
    }

    showTypingIndicator() {
        if (this.isTyping) return;
        
        this.isTyping = true;
        const messagesContainer = document.getElementById('chat-messages');
        
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message ai';
        typingDiv.id = 'typing-indicator';
        
        const bubbleDiv = document.createElement('div');
        bubbleDiv.className = 'typing-indicator';
        
        const dotsDiv = document.createElement('div');
        dotsDiv.className = 'typing-dots';
        
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('div');
            dot.className = 'typing-dot';
            dotsDiv.appendChild(dot);
        }
        
        bubbleDiv.appendChild(dotsDiv);
        typingDiv.appendChild(bubbleDiv);
        messagesContainer.appendChild(typingDiv);
        
        this.scrollToBottom();
    }

    hideTypingIndicator() {
        this.isTyping = false;
        const indicator = document.getElementById('typing-indicator');
        if (indicator) {
            indicator.remove();
        }
    }

    scrollToBottom() {
        const messagesContainer = document.getElementById('chat-messages');
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    // Replace simulateQueryAPI with live backend
    async simulateQueryAPI(query) {
        const token = localStorage.getItem('bioChat_token');
        const res = await fetch(`${API_BASE}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({ query })
        });
        if (!res.ok) {
            const err = await res.text().catch(() => '');
            throw new Error(`Query failed: ${res.status} ${err}`);
        }
        const data = await res.json();
        return {
            answer: data.answer,
            confidence_score: data.confidence_score,
            processing_time: data.processing_time,
            citations: data.citations,
            follow_up_questions: data.follow_up_questions,
            query_analysis: data.query_analysis
        };
    }

    extractEntities(query) {
        const entities = [];
        const biomedicalTerms = [
            'CAR-T', 'mRNA', 'vaccine', 'COVID-19', 'Alzheimer', 'CRISPR', 'gene editing',
            'protein', 'antibody', 'therapy', 'treatment', 'disease', 'pathology',
            'mechanism', 'clinical', 'trial', 'drug', 'biomarker'
        ];

        biomedicalTerms.forEach(term => {
            if (query.toLowerCase().includes(term.toLowerCase())) {
                entities.push(term);
            }
        });

        return entities.slice(0, 3); // Return max 3 entities
    }

    updateChatHistory(question, answer) {
        const historyItem = {
            id: Date.now(),
            title: question.substring(0, 50) + (question.length > 50 ? '...' : ''),
            preview: answer.substring(0, 100) + (answer.length > 100 ? '...' : ''),
            timestamp: new Date(),
            messages: [...this.currentMessages]
        };

        this.chatHistory.unshift(historyItem);
        
        // Keep only last 10 conversations
        if (this.chatHistory.length > 10) {
            this.chatHistory = this.chatHistory.slice(0, 10);
        }

        this.saveChatHistory();
        this.renderChatHistory();
    }

    loadChatHistory() {
        const saved = localStorage.getItem('bioChat_history');
        if (saved) {
            this.chatHistory = JSON.parse(saved);
        }
    }

    saveChatHistory() {
        localStorage.setItem('bioChat_history', JSON.stringify(this.chatHistory));
    }

    renderChatHistory() {
        const historyList = document.getElementById('history-list');
        historyList.innerHTML = '';

        this.chatHistory.forEach(item => {
            const historyDiv = document.createElement('div');
            historyDiv.className = 'history-item';
            historyDiv.onclick = () => this.loadChatFromHistory(item);

            const titleDiv = document.createElement('div');
            titleDiv.className = 'history-title-text';
            titleDiv.textContent = item.title;

            const previewDiv = document.createElement('div');
            previewDiv.className = 'history-preview';
            previewDiv.textContent = item.preview;

            historyDiv.appendChild(titleDiv);
            historyDiv.appendChild(previewDiv);
            historyList.appendChild(historyDiv);
        });
    }

    loadChatFromHistory(historyItem) {
        const messagesContainer = document.getElementById('chat-messages');
        messagesContainer.innerHTML = '';
        
        this.currentMessages = [...historyItem.messages];
        
        this.currentMessages.forEach(msg => {
            this.addMessage(msg.type, msg.content, msg.metadata);
        });

        // Close sidebar on mobile
        if (window.innerWidth <= 768) {
            this.closeSidebar();
        }
    }

    startNewChat() {
        const messagesContainer = document.getElementById('chat-messages');
        messagesContainer.innerHTML = `
            <div class="welcome-message">
                <div class="welcome-icon">🧬</div>
                <h3>Welcome to BioChat AI</h3>
                <p>Ask me anything about biomedical research, clinical studies, drug mechanisms, disease pathology, and more. I'll search through millions of PubMed articles to provide evidence-based answers with proper citations.</p>
                <div class="suggested-queries">
                    <h4>Try these sample queries:</h4>
                    <div class="query-suggestions">
                        <button class="query-suggestion" data-query="What are the mechanisms of CAR-T cell therapy?">
                            CAR-T cell therapy mechanisms
                        </button>
                        <button class="query-suggestion" data-query="How do mRNA vaccines work against COVID-19?">
                            mRNA vaccine mechanisms
                        </button>
                        <button class="query-suggestion" data-query="What causes Alzheimer's disease pathology?">
                            Alzheimer's disease pathology
                        </button>
                        <button class="query-suggestion" data-query="Explain CRISPR gene editing applications">
                            CRISPR gene editing applications
                        </button>
                    </div>
                </div>
            </div>
        `;
        
        this.currentMessages = [];
        
        // Close sidebar on mobile
        if (window.innerWidth <= 768) {
            this.closeSidebar();
        }
    }

    toggleSidebar() {
        const sidebar = document.getElementById('sidebar');
        this.sidebarOpen = !this.sidebarOpen;
        
        if (this.sidebarOpen) {
            sidebar.classList.add('open');
        } else {
            sidebar.classList.remove('open');
        }
    }

    closeSidebar() {
        const sidebar = document.getElementById('sidebar');
        sidebar.classList.remove('open');
        this.sidebarOpen = false;
    }

    logout() {
        localStorage.removeItem('bioChat_token');
        localStorage.removeItem('bioChat_user');
        this.currentUser = null;
        this.currentMessages = [];
        this.showAuthPage();
        
        // Reset form
        document.getElementById('auth-form').reset();
        this.switchAuthTab('login');
    }

    // Handle responsive design
    handleResize() {
        if (window.innerWidth > 768) {
            const sidebar = document.getElementById('sidebar');
            sidebar.classList.remove('open');
            this.sidebarOpen = false;
        }
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    const app = new BioChatApp();
    
    // Handle window resize
    window.addEventListener('resize', () => app.handleResize());
    
    // Handle clicks outside sidebar on mobile
    document.addEventListener('click', (e) => {
        const sidebar = document.getElementById('sidebar');
        const toggle = document.getElementById('toggle-sidebar');
        
        if (window.innerWidth <= 768 && 
            app.sidebarOpen && 
            !sidebar.contains(e.target) && 
            !toggle.contains(e.target)) {
            app.closeSidebar();
        }
    });
});

