import { Outlet, Link, createRootRoute, HeadContent, Scripts } from "@tanstack/react-router";
import { Toaster } from "@/components/ui/sonner";
import appCss from "../styles.css?url";

function NotFoundComponent() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-4">
      <div className="max-w-md text-center">
        <h1 className="text-7xl font-bold text-foreground">404</h1>
        <h2 className="mt-4 text-xl font-semibold text-foreground">Page not found</h2>
        <p className="mt-2 text-sm text-muted-foreground">
          The page you're looking for doesn't exist or has been moved.
        </p>
        <div className="mt-6">
          <Link
            to="/"
            className="inline-flex items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
          >
            Go home
          </Link>
        </div>
      </div>
    </div>
  );
}

export const Route = createRootRoute({
  head: () => ({
    meta: [
      { charSet: "utf-8" },
      { name: "viewport", content: "width=device-width, initial-scale=1" },
      { title: "Coraçai — AI Cardiomegaly Detection from Chest X-rays" },
      {
        name: "description",
        content:
          "Upload chest X-ray images and get an AI-assisted cardiomegaly probability score in seconds. Designed for clinicians and researchers.",
      },
      { name: "author", content: "Coraçai" },
      { property: "og:title", content: "Coraçai — AI Cardiomegaly Detection from Chest X-rays" },
      { property: "og:description", content: "Upload and classify medical images like X-rays with an intuitive UI." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
      { name: "twitter:title", content: "Coraçai — AI Cardiomegaly Detection from Chest X-rays" },
      { name: "description", content: "Upload and classify medical images like X-rays with an intuitive UI." },
      { name: "twitter:description", content: "Upload and classify medical images like X-rays with an intuitive UI." },
      { property: "og:image", content: "https://pub-bb2e103a32db4e198524a2e9ed8f35b4.r2.dev/d1564526-1b6b-4a50-8de2-78ab6de68c22/id-preview-1b66345b--43079148-fa84-4473-9c38-00159e4cfc53.lovable.app-1776862541619.png" },
      { name: "twitter:image", content: "https://pub-bb2e103a32db4e198524a2e9ed8f35b4.r2.dev/d1564526-1b6b-4a50-8de2-78ab6de68c22/id-preview-1b66345b--43079148-fa84-4473-9c38-00159e4cfc53.lovable.app-1776862541619.png" },
    ],
    links: [
      { rel: "stylesheet", href: appCss },
      { rel: "icon", type: "image/png", href: "/favicon.png" },
      { rel: "apple-touch-icon", href: "/favicon.png" },
    ],
  }),
  shellComponent: RootShell,
  component: RootComponent,
  notFoundComponent: NotFoundComponent,
});

function ThemeBootstrap() {
  // Avoid flash of incorrect theme by reading localStorage before hydration
  const code = `(function(){try{var t=localStorage.getItem('rx-theme');var d=t?t==='dark':window.matchMedia('(prefers-color-scheme: dark)').matches;if(d)document.documentElement.classList.add('dark');}catch(e){}})();`;
  return <script dangerouslySetInnerHTML={{ __html: code }} />;
}

function RootShell({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <HeadContent />
      </head>
      <body>
        <ThemeBootstrap />
        {children}
        <Toaster richColors position="top-right" />
        <Scripts />
      </body>
    </html>
  );
}

function RootComponent() {
  return <Outlet />;
}
