
export default function GazeDot(props){
  
  const style = {
    position: 'fixed',
    zIndex: 100,
    left: '-5px',
    top: '-5px',
    background: 'magenta',
    borderRadius: '50%',
    opacity: 0.7,
    width: 30,
    height: 30,
    transform: `translate(${props.x}px, ${props.y}px)`,
  };

  return <div className="z-100" id="GazeDot" style={style}></div>;
};