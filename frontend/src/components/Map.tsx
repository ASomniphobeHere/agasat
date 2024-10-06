"use client";

import {
  ImageOverlay,
  LayersControl,
  MapContainer,
  MapContainerProps,
  Marker,
  Popup,
  Rectangle,
  TileLayer,
  WMSTileLayer,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";
import "leaflet-defaulticon-compatibility";
import "leaflet-defaulticon-compatibility/dist/leaflet-defaulticon-compatibility.css";
import { useEffect } from "react";
import { useStore } from "@/hooks/state/store";
import {
  ForwardRefExoticComponent,
  LegacyRef,
  RefAttributes,
  useCallback,
  useMemo,
  useState,
} from "react";
import { Icon, type Map } from "leaflet";

// Sentinel Hub WMS service
// tiles generated using EPSG:3857 projection - Leaflet takes care of that
const baseUrl = `https://sh.dataspace.copernicus.eu/ogc/wms/${process.env.NEXT_PUBLIC_SENTINEL_HUB_TOKEN}`;

function createWmsLayer(layerId: string) {
  const layer = (
    <WMSTileLayer
      url={baseUrl}
      layers={layerId}
      tileSize={512}
      attribution={
        '&copy; <a href="https://dataspace.copernicus.eu/" target="_blank">Copernicus Data Space Ecosystem</a>'
      }
      minZoom={6}
      maxZoom={16}
      key={layerId}
    />
  );
  return layer;
}

const ndvi = createWmsLayer("NDVI");
const trueColor = createWmsLayer("TRUE_COLOR");
const urbanAreas = createWmsLayer("URBAN_AREAS");
const sentinel2cloudless = createWmsLayer("SENTINEL-2-CLOUDLESS");

// OpenStreetMap
const osm = (
  <TileLayer
    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
  />
);

const stadiaDark = (
  <TileLayer
    attribution='&copy; <a href="https://www.stadiamaps.com/" target="_blank">Stadia Maps</a> &copy; <a href="https://openmaptiles.org/" target="_blank">OpenMapTiles</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    url="https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png"
  />
);

const stadiaOrthoPhoto = (
  <TileLayer
    attribution='&copy; CNES, Distribution Airbus DS, © Airbus DS, © PlanetObserver (Contains Copernicus Data) | &copy; <a href="https://www.stadiamaps.com/" target="_blank">Stadia Maps</a> &copy; <a href="https://openmaptiles.org/" target="_blank">OpenMapTiles</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    url="https://tiles.stadiamaps.com/tiles/alidade_satellite/{z}/{x}/{y}{r}.jpg"
  />
);

const baseMaps = {
  OpenStreetMap: osm,
  StadiaDark: stadiaDark,
  StadiaOrthoPhoto: stadiaOrthoPhoto,
};

const overlayMaps = {
  NDVI: ndvi,
  "True Color": trueColor,
  //   "Urban Area": urbanAreas,
  //   "Sentinel-2 Cloudless Mosaic": sentinel2cloudless,
};

const customIcon = new Icon.Default({
  iconSize: [30, 45],
  iconAnchor: [10, 41],
  
  iconUrl: "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGNsYXNzPSJsdWNpZGUgbHVjaWRlLW1hcC1waW4iPjxwYXRoIGQ9Ik0yMCAxMGMwIDQuOTkzLTUuNTM5IDEwLjE5My03LjM5OSAxMS43OTlhMSAxIDAgMCAxLTEuMjAyIDBDOS41MzkgMjAuMTkzIDQgMTQuOTkzIDQgMTBhOCA4IDAgMCAxIDE2IDAiLz48Y2lyY2xlIGN4PSIxMiIgY3k9IjEwIiByPSIzIi8+PC9zdmc+",
});


export default function MapComponent() {
  const { mapImages, markers } = useStore();

  const [map, setMap] = useState<Map | null>(null);

  const { setBounds } = useStore();

  const onMove = useCallback(() => {
    if (map) {
      setBounds(map.getBounds());
    }
    // console.log(map?.getBounds());
  }, [map, setBounds]);

  useEffect(() => {
    if (map) {
      map.on("move", onMove);
      return () => {
        map.off("move", onMove);
      };
    }
  }, [map, onMove]);

  const displayMap = useMemo(
    () => (
      <MapContainer
        center={[56.946285, 24.105078]}
        zoom={13}
        scrollWheelZoom={true}
        className="w-full h-full"
        ref={setMap}
      >
        {/* {baseMaps.OpenStreetMap} */}
        {/* {baseMaps.StadiaDark} */}
        {baseMaps.StadiaOrthoPhoto}
        {/* <LayersControl>
        {Object.entries(overlayMaps).map((entry) => (
          <LayersControl.Overlay name={entry[0]} key={entry[0]} checked={false}>
            {entry[1]}
          </LayersControl.Overlay>
        ))}
      </LayersControl>
{/*  */}
        <LayersControl>
          {mapImages.map((image) => {
            console.log(image);
            return (
              <>
                <ImageOverlay
                  url={image.url}
                  bounds={image.bounds}
                  key={image.url}
                  opacity={1}
                />
                {/* <Rectangle bounds={image.bounds} /> */}
              </>
            );
          })}
        </LayersControl>
        {markers.map((marker) => {
          // console.log(marker);
          return (
            <Marker position={marker.coords} key={marker.coords.toString()} icon={customIcon}>
              <Popup>
                <div>
                  <p>{Array.isArray(marker.coords) ? marker.coords[0] : marker.coords.lat}</p>
                  <p>{Array.isArray(marker.coords) ? marker.coords[1] : marker.coords.lng}</p>
                </div>
              </Popup>
            </Marker>
          );
        })}
        <Marker position={[56.69, 26.56]} icon={customIcon}>
          <Popup>
            A pretty CSS3 popup. <br /> Easily customizable
          </Popup>
        </Marker>
      </MapContainer>
    ),
    [mapImages, markers]
  );

  return displayMap;
}
