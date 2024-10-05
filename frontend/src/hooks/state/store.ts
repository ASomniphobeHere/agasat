import { LatLngBoundsExpression, LatLngExpression } from "leaflet";
import { create } from "zustand";

interface MapImage {
    url: string
    bounds: LatLngBoundsExpression
}

interface Markers {
    coords: LatLngExpression
    label: string
    description: string
}

interface Store {
    mapImages:  MapImage[]
    setMapImages: (state: MapImage[]) => void
    markers: Markers[]
    setMarkers: (state: Markers[]) => void
}

export const useStore = create<Store>((set) => ({
  mapImages: [],
  setMapImages: (newImages: MapImage[]) => set((oldState) => ({ ...oldState, mapImages: newImages })),
  markers: [],
  setMarkers: (newMarkers: Markers[]) => set((oldState) => ({ ...oldState, markers: newMarkers })),
}));
