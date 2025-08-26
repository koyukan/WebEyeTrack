import React from "react";
import { Menu, Github } from "lucide-react";
import {
  NavigationMenu,
  NavigationMenuItem,
  NavigationMenuLink,
  NavigationMenuList,
} from "./components/ui/navigation-menu";
import { Button } from "./components/ui/button";
import { Sheet, SheetContent, SheetTrigger } from "./components/ui/sheet";
import { Separator } from "./components/ui/separator";

// If you're using Vite/CRA asset imports, replace with your logo path or remove
// import logo from "./logo.svg";

/**
 * Single-page App scaffold for the WebEyeTrack site
 * - Top fixed navbar built with shadcn/ui
 * - Smooth in-page scrolling to sections
 * - Mobile menu via <Sheet>
 * - Accessible, keyboard-friendly nav
 */

const SECTIONS = [
  { id: "overview", label: "Overview" },
  { id: "installation", label: "Installation" },
  { id: "usage", label: "Usage" },
  { id: "demo", label: "Demo" },
  { id: "publications", label: "Publications" },
  { id: "team", label: "Our Team" },
  { id: "acknowledgements", label: "Acknowledgements" },
  { id: "licensing", label: "Licensing" },
] as const;

function scrollToId(id: string) {
  const el = document.getElementById(id);
  if (!el) return;
  el.scrollIntoView({ behavior: "smooth", block: "start" });
}

function DesktopNav() {
  return (
    <NavigationMenu className="hidden md:flex">
      <NavigationMenuList>
        {SECTIONS.map((s) => (
          <NavigationMenuItem key={s.id}>
            <NavigationMenuLink asChild>
              <Button
                variant="ghost"
                size="sm"
                className="font-medium"
                onClick={() => scrollToId(s.id)}
              >
                {s.label}
              </Button>
            </NavigationMenuLink>
          </NavigationMenuItem>
        ))}
      </NavigationMenuList>
    </NavigationMenu>
  );
}

function MobileNav() {
  return (
    <Sheet>
      <SheetTrigger asChild>
        <Button variant="ghost" size="icon" className="md:hidden">
          <Menu className="h-5 w-5" />
          <span className="sr-only">Open menu</span>
        </Button>
      </SheetTrigger>
      <SheetContent side="left" className="w-72">
        <div className="mt-6 space-y-2">
          {SECTIONS.map((s) => (
            <Button
              key={s.id}
              variant="ghost"
              className="w-full justify-start text-base"
              onClick={() => {
                scrollToId(s.id);
                // Close the sheet by clicking the overlay programmatically
                const overlay = document.querySelector<HTMLElement>(
                  "[data-state='open'][data-radix-sheet-override]"
                );
                overlay?.click();
              }}
            >
              {s.label}
            </Button>
          ))}
        </div>
      </SheetContent>
    </Sheet>
  );
}

function Section({ id, title, children }: React.PropsWithChildren<{ id: string; title: string }>) {
  return (
    <section
      id={id}
      className="scroll-mt-24 md:scroll-mt-28 min-h-[45vh] py-12 md:py-20"
      aria-label={title}
    >
      <div className="container mx-auto max-w-5xl px-4">
        <h2 className="text-3xl md:text-4xl font-semibold tracking-tight">{title}</h2>
        <Separator className="my-6" />
        <div className="prose prose-zinc dark:prose-invert max-w-none">
          {children}
        </div>
      </div>
    </section>
  );
}

export default function App() {
  return (
    <div className="min-h-screen bg-background text-foreground w-full">
      {/* Fixed top bar */}
      <header className="sticky top-0 z-50 w-full border-b bg-background/80 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container mx-auto max-w-6xl px-4">
          <div className="flex h-16 md:h-20 items-center justify-between gap-3">
            {/* Brand */}
            <div className="flex items-center gap-3">
              {/* Uncomment if you have a logo asset */}
              {/* <img src={logo} alt="WebEyeTrack logo" className="h-7 w-7" /> */}
              <a
                href="#overview"
                onClick={(e) => {
                  e.preventDefault();
                  scrollToId("overview");
                }}
                className="text-lg md:text-xl font-bold"
              >
                WebEyeTrack
              </a>
            </div>

            <DesktopNav />
            <div className="flex items-center gap-2">
              <a
                href="https://github.com/RedForestAI/WebEyeTrack"
                target="_blank"
                rel="noreferrer noopener"
                className="hidden sm:inline-flex"
              >
                <Button variant="outline" size="sm" className="gap-2">
                  <Github className="h-4 w-4" />
                  GitHub
                </Button>
              </a>
              <MobileNav />
            </div>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main>
        <Section id="overview" title="Overview">
          <p>
            WebEyeTrack is a lightweight, web-native eye tracking toolkit. Use this
            space to describe the motivation, core ideas, and a visual teaser.
          </p>
        </Section>

        <Section id="installation" title="Installation">
          <pre>
          {`# via npm
          npm install webeyetrack

          # or yarn
          yarn add webeyetrack`}
          </pre>
          <p>
            Include any build/runtime prerequisites here.
          </p>
        </Section>

        <Section id="usage" title="Usage">
          <pre>
          {`import { EyeTracker } from 'webeyetrack'

          const tracker = new EyeTracker({ /* options */ })
          tracker.start()`}
          </pre>
        </Section>

        <Section id="demo" title="Demo">
          <p>
            Embed a live demo, gifs, or screenshots. You can also link to a demo
            route within your app.
          </p>
        </Section>

        <Section id="publications" title="Publications">
          <ul>
            <li>
              Doe, J., et al. <em>WebEyeTrack: Browser‑based Eye Tracking</em>, 2025.
            </li>
          </ul>
        </Section>

        <Section id="team" title="Our Team">
          <p>
            Introduce the authors and contributors. Add avatars/cards if you like.
          </p>
        </Section>

        <Section id="acknowledgements" title="Acknowledgements">
          <p>
            Credit funding sources, collaborators, and any third‑party libraries.
          </p>
        </Section>

        <Section id="licensing" title="Licensing">
          <p>
            Released under the MIT License (or your license of choice).
          </p>
        </Section>
      </main>

      {/* Footer */}
      <footer className="border-t py-8">
        <div className="container mx-auto max-w-6xl px-4 text-sm text-muted-foreground">
          © {new Date().getFullYear()} WebEyeTrack. All rights reserved.
        </div>
      </footer>

      {/* Back to top */}
      <div className="fixed bottom-5 right-5">
        <Button
          variant="secondary"
          onClick={() => scrollToId("overview")}
          className="shadow-lg"
        >
          Back to top
        </Button>
      </div>
    </div>
  );
}
