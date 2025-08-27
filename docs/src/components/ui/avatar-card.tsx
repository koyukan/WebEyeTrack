import React from "react";
import clsx from "clsx";
import { User } from "lucide-react";

export type AvatarCardProps = {
  /** Person's display name */
  name: string;
  /** Link to bio/profile page */
  href: string;
  /** Optional role or short subtitle */
  subtitle?: string;
  /** Image URL (can be relative) */
  imageSrc?: string;
  /** alt text for the image */
  imageAlt?: string;
  /** Render a short description below the name */
  children?: React.ReactNode;
  /** Open link in a new tab */
  external?: boolean;
  /** Avatar size (circle diameter) */
  size?: "sm" | "md" | "lg";
  className?: string;
};

function initialsFromName(name: string) {
  const parts = name.trim().split(/\s+/).slice(0, 2);
  return parts.map((p) => p[0]?.toUpperCase() ?? "").join("");
}

export function AvatarCard({
  name,
  href,
  subtitle,
  imageSrc,
  imageAlt,
  children,
  external,
  size = "md",
  className,
}: AvatarCardProps) {
  const [imgError, setImgError] = React.useState(false);

  const sizeClasses = {
    sm: "h-12 w-12 text-xs",
    md: "h-16 w-16 text-sm",
    lg: "h-20 w-20 text-base",
  } as const;

  const linkProps = external
    ? { target: "_blank", rel: "noopener noreferrer" }
    : {};

  return (
    <div
      className={clsx(
        "group relative overflow-hidden rounded-xl border bg-card text-card-foreground shadow-sm",
        "transition hover:shadow-md focus-within:shadow-md",
        className
      )}
    >
      <div className="flex items-start gap-4 p-4 sm:p-5">
        {/* Avatar */}
        <div className="shrink-0">
          {imageSrc && !imgError ? (
            <img
              src={imageSrc}
              alt={imageAlt ?? name}
              onError={() => setImgError(true)}
              className={clsx(
                "rounded-full object-cover bg-muted border border-border",
                sizeClasses[size]
              )}
            />
          ) : (
            <div
              aria-hidden
              className={clsx(
                "grid place-items-center rounded-full bg-muted text-muted-foreground border border-border",
                sizeClasses[size]
              )}
            >
              {name ? (
                <span className="font-medium">{initialsFromName(name)}</span>
              ) : (
                <User className="h-5 w-5" />
              )}
            </div>
          )}
        </div>

        {/* Text */}
        <div className="min-w-0">
          <h3 className="text-base font-semibold leading-tight tracking-tight">
            <a
              href={href}
              {...linkProps}
              className="focus:outline-none focus-visible:ring-2 focus-visible:ring-ring/60 focus-visible:ring-offset-2 rounded-sm hover:underline"
            >
              {name}
            </a>
          </h3>
          {subtitle && (
            <p className="mt-0.5 text-sm text-muted-foreground">{subtitle}</p>
          )}
          {children && (
            <div className="mt-2 text-sm leading-relaxed text-muted-foreground/90">
              {children}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default AvatarCard;