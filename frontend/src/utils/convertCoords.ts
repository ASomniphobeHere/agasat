import type { LatLngBoundsExpression } from "leaflet";
export function convertLatLngBoundsToNWSE(bounds: LatLngBoundsExpression) {
  if (Array.isArray(bounds) && bounds.length === 2 && Array.isArray(bounds[0]) && Array.isArray(bounds[1])) {
    return [[bounds[1][0], bounds[0][1]], [bounds[0][0], bounds[1][1]]];
  } else if (typeof bounds === 'object' && bounds !== null && 'getNorthWest' in bounds && 'getSouthEast' in bounds) {
    const nw = bounds.getNorthWest();
    const se = bounds.getSouthEast();
    return [[nw.lat, nw.lng], [se.lat, se.lng]];
  }
  return bounds;
}

// export function convertNWSEToLatLngBounds(nwseBounds: LatLngBoundsExpression) {
//   if (Array.isArray(nwseBounds) && nwseBounds.length === 2 && Array.isArray(nwseBounds[0]) && Array.isArray(nwseBounds[1])) {
//     return [[nwseBounds[1][0], nwseBounds[0][1]], [nwseBounds[0][0], nwseBounds[1][1]]];
//   } else if (typeof nwseBounds === 'object' && nwseBounds !== null && 'getSouthWest' in nwseBounds && 'getNorthEast' in nwseBounds) {
//     const sw = nwseBounds.getSouthWest();
//     const ne = nwseBounds.getNorthEast();
//     return [[sw.lat, ne.lng], [ne.lat, sw.lng]];
//   }
//   return nwseBounds;
// }
