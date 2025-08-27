import React, { type JSX } from "react";
import clsx from "clsx";

export interface InlineCodeProps
  extends React.HTMLAttributes<HTMLElement> {
  /** Which element to render (defaults to <code>) */
  as?: "code" | "span";
}

/**
 * InlineCode — a tiny, shadcn‑style inline tag for variables, package names, etc.
 * Safe to place inside <p>, lists, headings, and within prose blocks.
 */
export function InlineCode({ as = "code", className, ...props }: InlineCodeProps) {
  const Tag = as as keyof JSX.IntrinsicElements;
  return (
    // @ts-ignore
    <Tag
      {...props}
      className={clsx(
        "inline-flex items-center align-middle rounded border border-border bg-muted",
        "px-1.5 py-0.5 font-mono text-[0.85em] leading-none text-foreground/90 shadow-sm",
        className
      )}
    />
  );
}

export default InlineCode;
