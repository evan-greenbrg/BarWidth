    # Start building river dem
    river_dem = np.full(dem.shape, 0.)
    for idx, row in coordinates.iterrows():
        x, y = georef.lonlat2pix(gmDEM, row['lon'], row['lat'])
        river_dem[y, x] = row['elev_0']

