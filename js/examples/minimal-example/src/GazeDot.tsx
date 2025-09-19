
type GazeDotProps = {
  x: number;
  y: number;
  gazeState: string;
}

export default function GazeDot(props: GazeDotProps){

  // Hide the dot is props.gazeState is 'closed'
  const style: React.CSSProperties = {
    position: 'fixed' as const,
    zIndex: 100,
    left: -5,
    top: -5,
    background: 'magenta',
    borderRadius: '50%',
    opacity: 0.7,
    width: 30,
    height: 30,
    display: props.gazeState === 'closed' ? 'none' : 'block',
    transform: `translate(${props.x}px, ${props.y}px)`,
  };

  return <div className="z-100" id="GazeDot" style={style}></div>;
};