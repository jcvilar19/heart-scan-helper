import { useEffect, useState } from "react";

type Theme = "light" | "dark";
const STORAGE_KEY = "rx-theme";

function hasDocument() {
  return typeof document !== "undefined";
}

function hasLocalStorage() {
  return typeof localStorage !== "undefined";
}

function getInitialTheme(): Theme {
  // Trust the class set by the pre-hydration ThemeBootstrap script first
  // so every hook instance agrees with the DOM.
  if (hasDocument() && document.documentElement.classList.contains("dark")) return "dark";

  if (hasLocalStorage()) {
    const stored = localStorage.getItem(STORAGE_KEY) as Theme | null;
    if (stored === "light" || stored === "dark") return stored;
  }

  if (typeof window !== "undefined" && typeof window.matchMedia === "function") {
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  return "light";
}

export function useTheme() {
  const [theme, setTheme] = useState<Theme>(getInitialTheme);

  useEffect(() => {
    if (!hasDocument()) return;
    document.documentElement.classList.toggle("dark", theme === "dark");
    if (hasLocalStorage()) {
      localStorage.setItem(STORAGE_KEY, theme);
    }
  }, [theme]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const onStorage = (e: StorageEvent) => {
      if (e.key === STORAGE_KEY && (e.newValue === "light" || e.newValue === "dark")) {
        setTheme(e.newValue);
      }
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  return {
    theme,
    toggle: () => setTheme((t) => (t === "dark" ? "light" : "dark")),
    setTheme,
  };
}
