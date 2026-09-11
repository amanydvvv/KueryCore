import React, { useState, useEffect } from 'react';
import Layout from './components/layout/Layout';
import ChatContainer from './components/chat/ChatContainer';
import AuthModal from './components/AuthModal';
import ResetPassword from './components/ResetPassword';
import WorkspaceSkeleton from './components/layout/WorkspaceSkeleton';
import { useConversations } from './hooks/useConversations';
import { getAuthToken, removeAuthToken, fetchCurrentUser, warmUpBackend } from './services/api';

function App() {
  const [user, setUser] = useState(null);
  const [checkingAuth, setCheckingAuth] = useState(true);
  const [isSigningIn, setIsSigningIn] = useState(false);

  // Detect /reset-password route without a full router dependency
  const isResetRoute = window.location.pathname === '/reset-password';

  // Immediately kick off background warm-up for Render free tier
  useEffect(() => {
    warmUpBackend();
  }, []);

  const {
    conversations,
    activeConversationId,
    messages,
    isLoadingHistory,
    isGenerating,
    error,
    clearError,
    selectConversation,
    startNewChat,
    removeConversation,
    sendMessage,
    loadConversationList,
  } = useConversations();

  useEffect(() => {
    // Skip auth check on the reset-password route — no token needed there
    if (isResetRoute) {
      setCheckingAuth(false);
      return;
    }

    async function initAuth() {
      const token = getAuthToken();
      if (!token) {
        setCheckingAuth(false);
        return;
      }
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3500);

      try {
        const u = await fetchCurrentUser(controller.signal);
        clearTimeout(timeoutId);
        setUser(u);
        if (loadConversationList) {
          loadConversationList().catch(() => {});
        }
      } catch (err) {
        clearTimeout(timeoutId);
        // Stale or timed-out token: clear and gracefully drop to sign-in modal
        removeAuthToken();
        setUser(null);
      } finally {
        setCheckingAuth(false);
      }
    }
    initAuth();

    const handleAuthExpired = () => setUser(null);
    window.addEventListener('auth-expired', handleAuthExpired);
    return () => window.removeEventListener('auth-expired', handleAuthExpired);
  }, []);

  // Fail-safe: Ensure skeleton screen NEVER stays stuck under any circumstance
  useEffect(() => {
    if (checkingAuth || isSigningIn) {
      const timer = setTimeout(() => {
        setCheckingAuth(false);
        setIsSigningIn(false);
      }, 3800);
      return () => clearTimeout(timer);
    }
  }, [checkingAuth, isSigningIn]);

  const handleAuthSuccess = (authData) => {
    // Show brief shimmer transition while user profile renders
    setIsSigningIn(true);
    setUser({ id: authData.user_id, email: authData.email });
    if (loadConversationList) {
      loadConversationList().catch(() => {});
    }
    setTimeout(() => {
      setIsSigningIn(false);
    }, 200);
  };

  // Called by AuthModal when auth fails — ensures skeleton never stays stuck
  const handleAuthError = () => {
    setIsSigningIn(false);
  };

  const handleLogout = () => {
    removeAuthToken();
    setUser(null);
  };

  // Navigate back to sign-in from the reset-password page
  const handleNavigateToSignIn = () => {
    window.history.replaceState(null, '', '/');
    window.location.reload(); // simplest SPA re-entry without a full router
  };

  const activeConversation = conversations.find((c) => c.id === activeConversationId);

  useEffect(() => {
    if (activeConversation?.title) {
      document.title = `${activeConversation.title} · KueryCore`;
    } else {
      document.title = 'KueryCore';
    }
  }, [activeConversation?.title]);

  // ── Reset password route ──────────────────────────────
  if (isResetRoute) {
    return <ResetPassword onNavigateToSignIn={handleNavigateToSignIn} />;
  }

  // ── Loading / skeleton ────────────────────────────────
  if (checkingAuth || isSigningIn) {
    return (
      <WorkspaceSkeleton
        onDismiss={() => {
          setCheckingAuth(false);
          setIsSigningIn(false);
        }}
      />
    );
  }

  // ── Auth gate ─────────────────────────────────────────
  if (!user) {
    return (
      <AuthModal
        onAuthSuccess={handleAuthSuccess}
        onAuthError={handleAuthError}
      />
    );
  }

  // ── Workspace ─────────────────────────────────────────
  return (
    <Layout
      user={user}
      onLogout={handleLogout}
      conversations={conversations}
      activeConversationId={activeConversationId}
      onSelectConversation={selectConversation}
      onStartNewChat={startNewChat}
      onDeleteConversation={removeConversation}
    >
      <ChatContainer
        activeConversation={activeConversation}
        messages={messages}
        isLoadingHistory={isLoadingHistory}
        isGenerating={isGenerating}
        error={error}
        onDismissError={clearError}
        onSendMessage={sendMessage}
      />
    </Layout>
  );
}

export default App;
