import type { Analysis, Point, RegionObservation } from './types'
export function regionViewport(landmarks:Point[],regionId:string,width:number,height:number): {x:number;y:number;width:number;height:number}
export function regionDrawing(result: Analysis, region: RegionObservation): {points: Point[]; contours: Point[][]; lines: {a: {x:number;y:number}; b:{x:number;y:number}; metric:string; label:string}[]}
