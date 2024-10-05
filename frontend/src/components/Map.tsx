"use client";

import {
  LayersControl,
  MapContainer,
  Marker,
  Popup,
  TileLayer,
  WMSTileLayer,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";
import "leaflet-defaulticon-compatibility";
import "leaflet-defaulticon-compatibility/dist/leaflet-defaulticon-compatibility.css";

// Sentinel Hub WMS service
// tiles generated using EPSG:3857 projection - Leaflet takes care of that
const baseUrl =
  "https://sh.dataspace.copernicus.eu/ogc/wms/df3eb751-18c3-46e9-824b-3b7de8634109";

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

const baseMaps = {
  OpenStreetMap: osm,
};

const overlayMaps = {
  NDVI: ndvi,
  "True Color": trueColor,
//   "Urban Area": urbanAreas,
//   "Sentinel-2 Cloudless Mosaic": sentinel2cloudless,
};

export default function Map() {
  return (
    <MapContainer
      center={[56.946285, 24.105078]}
      zoom={13}
      scrollWheelZoom={true}
      className="w-full h-full"
    >
      {baseMaps.OpenStreetMap}
      <LayersControl>
        {Object.entries(overlayMaps).map((entry) => (
          <LayersControl.Overlay name={entry[0]} key={entry[0]} checked={false}>
            {entry[1]}
          </LayersControl.Overlay>
        ))}
      </LayersControl>

      <Marker position={[51.505, -0.09]}>
        <Popup>
          A pretty CSS3 popup. <br /> Easily customizable.
        </Popup>
      </Marker>
    </MapContainer>
  );
}
