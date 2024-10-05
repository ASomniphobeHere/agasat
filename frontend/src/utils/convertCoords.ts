import type { LatLngBoundsExpression } from "leaflet";
export function convertLatLngBoundsToNWSE(bounds: LatLngBoundsExpression) {
  if (Array.isArray(bounds) && bounds.length === 2 && Array.isArray(bounds[0]) && Array.isArray(bounds[1])) {
    return [[bounds[1][0], bounds[0][1]], [bounds[0][0], bounds[1][1]]];
  } else if (typeof bounds === 'object' && bounds !== null && 'getSouthWest' in bounds && 'getNorthEast' in bounds) {
    const sw = bounds.getSouthWest();
    const ne = bounds.getNorthEast();
    return [[ne.lat, sw.lng], [sw.lat, ne.lng]];
  }
  return bounds;
}
