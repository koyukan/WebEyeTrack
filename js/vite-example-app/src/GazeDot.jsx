import React from 'react';

export default function GazeDot(x, y){
  
  const style = {
    position: 'fixed',
    zIndex: 100,
    left: '-5px',
    top: '-5px',
    background: 'red',
    borderRadius: '50%',
    opacity: 0.7,
    width: 20,
    height: 20,
    transform: `translate(${x}px, ${y}px)`,
  };

  return <div id="GazeDot" style={style}></div>;
};