import fs from 'fs';

const parseContours = (data: string): number[][][] => {
    const contours: number[][][] = [];
    const regex = /\[\[\[(.*?)\]\]\]/g;
    let match;
    while ((match = regex.exec(data)) !== null) {
        const contour = match[1]
            .split('],[')
            .map(pair => pair.replace(/[\[\]]/g, '').split(' ').map(Number));
        contours.push(contour);
    }
    return contours;
};

const data = fs.readFileSync('server/contours.txt', 'utf-8');
const contours = parseContours(data);
console.log(contours);