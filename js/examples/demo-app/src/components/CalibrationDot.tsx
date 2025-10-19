/**
 * CalibrationDot Component
 *
 * Displays an animated calibration dot with crosshair overlay
 * Reference: Python implementation at python/demo/calibration_widget.py
 *
 * Features:
 * - Circular dot with perpendicular crosshair lines
 * - Color animation: red → white (2000ms) to guide user attention
 * - Positioned using normalized coordinates [-0.5, 0.5]
 */

import React, { useEffect, useState } from 'react';
import { CalibrationPoint } from '../types/calibration';
import { normalizedToPixels } from '../utils/calibrationHelpers';

interface CalibrationDotProps {
  /** Position in normalized coordinates [-0.5, 0.5] */
  position: CalibrationPoint;

  /** Duration of color animation in milliseconds (default: 2000) */
  animationDuration?: number;

  /** Callback when animation completes (dot turns white) */
  onAnimationComplete?: () => void;

  /** Size of the dot in pixels (default: 40) */
  size?: number;
}

export default function CalibrationDot({
  position,
  animationDuration = 2000,
  onAnimationComplete,
  size = 40
}: CalibrationDotProps) {
  const [isWhite, setIsWhite] = useState(false);

  // Convert normalized position to pixel coordinates
  const pixelPosition = normalizedToPixels(
    position,
    window.innerWidth,
    window.innerHeight
  );

  // Trigger animation on mount or position change
  useEffect(() => {
    setIsWhite(false);

    // Start animation after brief delay to ensure reset is visible
    const startTimer = setTimeout(() => {
      setIsWhite(true);
    }, 50);

    // Trigger completion callback when animation finishes
    const completeTimer = setTimeout(() => {
      if (onAnimationComplete) {
        onAnimationComplete();
      }
    }, animationDuration + 50);

    return () => {
      clearTimeout(startTimer);
      clearTimeout(completeTimer);
    };
  }, [position, animationDuration, onAnimationComplete]);

  const crosshairLength = size * 1.5;
  const crosshairThickness = 2;

  return (
    <div
      className="absolute pointer-events-none"
      style={{
        left: pixelPosition.x,
        top: pixelPosition.y,
        transform: 'translate(-50%, -50%)', // Center the dot on the position
      }}
    >
      {/* Main circular dot */}
      <div
        className="rounded-full transition-colors"
        style={{
          width: size,
          height: size,
          backgroundColor: isWhite ? '#ffffff' : '#ff0000',
          transitionDuration: `${animationDuration}ms`,
          transitionTimingFunction: 'linear',
          boxShadow: '0 0 10px rgba(0, 0, 0, 0.5)', // Add subtle shadow for visibility
        }}
      />

      {/* Crosshair overlay - Horizontal line */}
      <div
        className="absolute bg-white"
        style={{
          left: '50%',
          top: '50%',
          width: crosshairLength,
          height: crosshairThickness,
          transform: 'translate(-50%, -50%)',
          opacity: 0.8,
        }}
      />

      {/* Crosshair overlay - Vertical line */}
      <div
        className="absolute bg-white"
        style={{
          left: '50%',
          top: '50%',
          width: crosshairThickness,
          height: crosshairLength,
          transform: 'translate(-50%, -50%)',
          opacity: 0.8,
        }}
      />
    </div>
  );
}
