import React from "react";
import clsx from "clsx";

export type YouTubeEmbedProps = {
  /** Full YouTube URL or bare video id */
  src: string;
  /** Accessible title for the iframe */
  title?: string;
  /** 16/9 by default. Provide a ratio like 16/9, 4/3, or a custom string (e.g., '56.25%'). */
  aspect?: "16/9" | "4/3" | "1/1" | string;
  /** Defer loading until scrolled into view */
  lazy?: boolean;
  /** Start time in seconds */
  start?: number;
  /** Auto-play after user interaction (applies on first load when clicking the thumbnail) */
  autoplay?: boolean;
  /** Hide related videos */
  rel?: 0 | 1;
  /** Reduce YouTube branding */
  modestBranding?: 0 | 1;
  /** Show player controls */
  controls?: 0 | 1;
  /** Additional class names on wrapper */
  className?: string;
  /** Render a lightweight click-to-play thumbnail instead of immediate iframe */
  lite?: boolean;
  /** Optional custom thumbnail URL; defaults to YouTube HQ thumbnail */
  thumbnailUrl?: string;
};

function extractId(src: string) {
  // Accept: youtu.be/<id>, youtube.com/watch?v=<id>, youtube.com/embed/<id>, or raw id
  const short = /youtu\.be\/(^[\n\r\s]+)?([\w-]{11})/;
  const watch = /v=([\w-]{11})/;
  const embed = /embed\/([\w-]{11})/;
  const raw = /^[\w-]{11}$/;
  if (raw.test(src)) return src;
  const s = src.toString();
  const m1 = s.match(watch);
  if (m1) return m1[1];
  const m2 = s.match(embed);
  if (m2) return m2[1];
  const m3 = s.match(short);
  if (m3) return m3[2];
  return src; // best effort
}

export function YouTubeEmbed({
  src,
  title = "YouTube video player",
  aspect = "16/9",
  lazy = true,
  start,
  autoplay = false,
  rel = 0,
  modestBranding = 1,
  controls = 1,
  className,
  lite = true,
  thumbnailUrl,
}: YouTubeEmbedProps) {
  const id = React.useMemo(() => extractId(src), [src]);

  const padTop = React.useMemo(() => {
    if (typeof aspect === "string" && aspect.includes("/")) {
      const [w, h] = aspect.split("/").map(Number);
      if (w && h) return `${(h / w) * 100}%`;
    }
    if (aspect.endsWith("%")) return aspect; // custom string like '56.25%'
    return "56.25%"; // default 16/9
  }, [aspect]);

  const params = new URLSearchParams({
    rel: String(rel),
    modestbranding: String(modestBranding),
    controls: String(controls),
    playsinline: "1",
  });
  if (start) params.set("start", String(start));
  if (autoplay) params.set("autoplay", "1");

  const embedUrl = `https://www.youtube-nocookie.com/embed/${id}?${params.toString()}`;
  const thumb =
    thumbnailUrl || `https://i.ytimg.com/vi_webp/${id}/hqdefault.webp`;

  const [showIframe, setShowIframe] = React.useState(!lite);

  return (
    <div
      className={clsx(
        "relative w-full overflow-hidden rounded-lg border bg-black",
        className
      )}
      style={{ paddingTop: padTop }}
    >
      {showIframe ? (
        <iframe
          className="absolute inset-0 h-full w-full"
          src={embedUrl}
          title={title}
          loading={lazy ? "lazy" : undefined}
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
          allowFullScreen
          referrerPolicy="strict-origin-when-cross-origin"
        />
      ) : (
        <button
          type="button"
          onClick={() => setShowIframe(true)}
          className={clsx(
            "absolute inset-0 grid place-items-center",
            "text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-ring/60 focus-visible:ring-offset-2"
          )}
          aria-label="Play video"
        >
          {/* Thumbnail background */}
          <img
            src={thumb}
            alt="Video thumbnail"
            className="absolute inset-0 h-full w-full object-cover opacity-90"
            loading={lazy ? "lazy" : undefined}
          />
          {/* Play button */}
          <span
            className="relative z-10 inline-block rounded-full bg-white/90 p-4 shadow-md transition hover:scale-105"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" fill="currentColor" viewBox="0 0 16 16" className="text-black">
              <path d="M11.596 8.697l-6.363 3.692A.75.75 0 0 1 4 11.742V4.258a.75.75 0 0 1 1.233-.647l6.363 3.692a.75.75 0 0 1 0 1.294z"/>
            </svg>
          </span>
        </button>
      )}
    </div>
  );
}

export default YouTubeEmbed;
