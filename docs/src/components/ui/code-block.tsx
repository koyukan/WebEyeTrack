import React from "react";
import clsx from "clsx";
import { Copy, Check } from "lucide-react";
import { Button } from "./button";

/**
 * Copy-to-clipboard utility with a DOM fallback.
 */
async function copyText(text: string) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    try {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.focus();
      ta.select();
      const ok = document.execCommand("copy");
      document.body.removeChild(ta);
      return ok;
    } catch {
      return false;
    }
  }
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = React.useState(false);

  async function onCopy() {
    const ok = await copyText(text);
    if (ok) {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1200);
    }
  }

  return (
    <Button
      variant="outline"
      size="icon"
      onClick={onCopy}
      aria-label="Copy code to clipboard"
      className="h-8 w-8"
      title="Copy"
    >
      {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
      <span className="sr-only">{copied ? "Copied" : "Copy"}</span>
    </Button>
  );
}

export type CodeBlockProps = {
  /** Raw code string */
  code: string;
  /** Language label shown in the header (no highlighting) */
  language?: string;
  /** Wrap long lines. Default false (horizontal scroll). */
  wrap?: boolean;
  className?: string;
};

/**
 * CodeBlock — shadcn-style code container with a header and copy button.
 * No syntax highlighting (keeps deps light). Add prism/shiki later if desired.
 */
export function CodeBlock({ code, language = "bash", wrap = false, className }: CodeBlockProps) {
  return (
    <div
      className={clsx(
        "group relative rounded-md border bg-muted/50 text-sm",
        className
      )}
    >
      <div className="flex items-center justify-between gap-2 border-b px-3 py-2">
        <span className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
          {language}
        </span>
        <CopyButton text={code} />
      </div>
      <div className="relative">
        <pre
          className={clsx(
            "overflow-x-auto p-4 font-mono leading-relaxed",
            wrap ? "whitespace-pre-wrap break-words" : "whitespace-pre"
          )}
        >
          <code>{code}</code>
        </pre>
      </div>
    </div>
  );
}

export default CodeBlock;
