import React from 'react';

interface DebugOverlayProps {
  data: Record<string, string | number | boolean | (number | string)[]>;
}

const DebugOverlay: React.FC<DebugOverlayProps> = ({ data }) => {
  return (
    <div className="fixed bottom-2 right-2 bg-black bg-opacity-70 text-white text-xs p-2 rounded shadow-md max-w-xs z-50 font-mono">
      {Object.entries(data).map(([key, value]) => (
        <div key={key}>
          <strong>{key}:</strong>{' '}
          {Array.isArray(value) ? value.join(', ') : String(value)}
        </div>
      ))}
    </div>
  );
};

export default DebugOverlay;
