import { Link, useNavigate } from "@tanstack/react-router";
import { Activity, History, LogIn, LogOut, Moon, Sun } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/hooks/use-auth";
import { useTheme } from "@/hooks/use-theme";
import { supabase } from "@/integrations/supabase/client";

export function AppHeader() {
  const { user } = useAuth();
  const { theme, toggle } = useTheme();
  const navigate = useNavigate();

  const handleSignOut = async () => {
    await supabase.auth.signOut();
    navigate({ to: "/" });
  };

  return (
    <header className="sticky top-0 z-40 w-full border-b border-border/60 bg-background/80 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-4 sm:px-6">
        <Link to="/" className="flex items-center gap-2.5 group">
          <span
            className="flex h-9 w-9 items-center justify-center rounded-lg text-primary-foreground shadow-[var(--shadow-elegant)] transition-transform group-hover:scale-105"
            style={{ background: "var(--gradient-primary)" }}
          >
            <Activity className="h-5 w-5" strokeWidth={2.5} />
          </span>
          <div className="flex flex-col leading-tight">
            <span className="text-sm font-semibold tracking-tight">CardioScan</span>
            <span className="text-[10px] uppercase tracking-widest text-muted-foreground">
              AI Radiology Assist
            </span>
          </div>
        </Link>

        <nav className="flex items-center gap-1 sm:gap-2">
          {user && (
            <Button asChild variant="ghost" size="sm">
              <Link to="/history">
                <History className="h-4 w-4" />
                <span className="hidden sm:inline">History</span>
              </Link>
            </Button>
          )}

          <Button
            variant="ghost"
            size="icon"
            onClick={toggle}
            aria-label="Toggle theme"
          >
            {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
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
