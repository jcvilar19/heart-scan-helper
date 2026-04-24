import { useEffect, useState } from "react";
import { Link, useNavigate } from "@tanstack/react-router";
import { Activity, History, LogIn, LogOut, Moon, Sun, UserRound } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/hooks/use-auth";
import { useTheme } from "@/hooks/use-theme";
import { supabase } from "@/integrations/supabase/client";
import logoUrl from "@/assets/logo.png";

export function AppHeader() {
  const { user } = useAuth();
  const { theme, toggle } = useTheme();
  const navigate = useNavigate();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const handleSignOut = async () => {
    await supabase.auth.signOut();
    navigate({ to: "/" });
  };

  return (
    <header className="sticky top-0 z-40 w-full border-b border-border/60 bg-background/80 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-4 sm:px-6">
        <Link to="/" className="flex items-center gap-2.5 group">
          <img
            src={logoUrl}
            alt="Coraçai logo"
            className="h-9 w-9 rounded-lg object-cover shadow-[var(--shadow-elegant)] transition-transform group-hover:scale-105"
          />
          <div className="flex flex-col leading-tight">
            <span className="text-sm font-semibold tracking-tight">Coraçai</span>
            <span className="text-[10px] uppercase tracking-widest text-muted-foreground">
              AI Radiology Assist
            </span>
          </div>
        </Link>

        <nav className="flex items-center gap-1 sm:gap-2">
          {user && (
            <>
              <Button asChild variant="ghost" size="sm">
                <Link to="/app">
                  <Activity className="h-4 w-4" />
                  <span className="hidden sm:inline">Scanner</span>
                </Link>
              </Button>
              <Button asChild variant="ghost" size="sm">
                <Link to="/history">
                  <History className="h-4 w-4" />
                  <span className="hidden sm:inline">History</span>
                </Link>
              </Button>
              <Button asChild variant="ghost" size="sm">
                <Link to="/profile">
                  <UserRound className="h-4 w-4" />
                  <span className="hidden sm:inline">Profile</span>
                </Link>
              </Button>
            </>
          )}

          <Button
            variant="ghost"
            size="icon"
            onClick={toggle}
            aria-label="Toggle theme"
          >
            {mounted && theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </Button>

          {user ? (
            <Button variant="outline" size="sm" onClick={handleSignOut}>
              <LogOut className="h-4 w-4" />
              <span className="hidden sm:inline">Sign out</span>
            </Button>
          ) : (
            <Button asChild size="sm">
              <Link to="/auth">
                <LogIn className="h-4 w-4" />
                <span>Sign in</span>
              </Link>
            </Button>
          )}
        </nav>
      </div>
    </header>
  );
}
