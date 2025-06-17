import React from 'react';

interface DebugOverlayProps {
  data: Record<string, string | number | boolean | (number | string)[]>;
}

const formatValue = (value: string | number | boolean): string => {
  if (typeof value === 'number') {
    return value.toFixed(3);
  }
  return String(value);
};

const DebugOverlay: React.FC<DebugOverlayProps> = ({ data }) => {
  return (
    <div className="fixed bottom-2 right-2 bg-black bg-opacity-70 text-white text-xs p-2 rounded shadow-md max-w-xs z-50 font-mono whitespace-pre">
      {Object.entries(data).map(([key, value]) => (
        <div key={key}>
          <strong>{key}:</strong>{' '}
          {Array.isArray(value)
            ? value.map(formatValue).join(', ')
            : formatValue(value)}
        </div>
      ))}
    </div>
  );
};

export default DebugOverlay;
