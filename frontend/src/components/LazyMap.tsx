'use client';

import dynamic from 'next/dynamic';

const LazyMapComponent = dynamic(() => import("@/components/Map"), {
  ssr: false,
  loading: () => <p>Loading...</p>,
});

function LazyMap() {
  return <LazyMapComponent />;
}

export default LazyMap;