import { Metadata } from "next";
import { useMemo } from "react";
import dynamic from "next/dynamic";

export const metadata: Metadata = {
  title: "Map",
};

export default function Home() {
  const Map = useMemo(
    () =>
      dynamic(() => import("../components/Map"), {
        loading: () => <p>A map is loading</p>,
        ssr: false,
      }),
    []
  );

  return (
    <div className="w-[80vw] h-[100vh]">
      <Map />
    </div>
  );
}
