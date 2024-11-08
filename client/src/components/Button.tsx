import React, { useState } from "react";

interface ButtonProps {
	text: string;
}

const Button: React.FC<ButtonProps> = ({ text }) => {
	const [count, setCount] = useState<number>(0);

	const handleClick = () => {
		setCount(count + 1);
	};

	return (
		<>
			<button onClick={handleClick}>{text}</button>
			<div>{count}</div>
		</>
	);
};

export default Button;
