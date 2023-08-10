 /*globals TextStream, File, MoleculePlugin, Molecule, Application, MessageBox, NMRPredictor, molecule, gc, Custom1DCsvConverter, performance*/


function predict_1H(smiles, out_path){

    // Create new document and predict spectrum
    dw = Application.mainWindow.newDocument();
    mol = get_mol_from_smiles(dw, smiles);

    // Predict NMR
    specId1H = NMRPredictor.predict(mol, "1H");
    var spec = nmr.activeSpectrum()	  

    // Process NMR
    process_1H_NMR(spec);

    // Multiplets
    var multiplets = spec.multiplets()
    
	// Save data
	var format = settings.value("Custom1DCsvConverter/CurrentFormat", "{ppm}{tab}{real}{tab}{imag}");
	save_spectrum_multiplet(smiles, dw.curPage()['items'], out_path, format.replace(/\{tab\}/g, ","), 6, false, multiplets)
    
	dw.destroy();
}

function process_1H_NMR(nmr_spectrum){
	mainWindow.doAction("nmrAutoPeakPicking")
	peaks = nmr_spectrum.peaks()	
	for (i=0; i<peaks.count; i++){
		peak = peaks.at(i)
		if (peak.type == Peak.Types.Solvent){
			peak.type = Peak.Types.Compound
			peak.flags = '2151682176'			
		}				
	}		
	mainWindow.doAction("nmrMultipletsAuto")
}


function get_mol_from_smiles(aDocWin, aSMILES){
    var mol_id = molecule.importSMILES(aSMILES);
    mol = new Molecule(aDocWin.getItem(mol_id))
    return mol
}

function save_spectrum_multiplet(smiles, aPageItems, aFilename, aFormat, aDecimals, aReverse, multiplets){
	"use strict";
	
	var file, strm, hz, dHz, i, pt, dPt, endPt, ppm, dPpm, spec,
		mapObj = {};
	try {
		file = new File(aFilename);	
		if (file.open(File.WriteOnly)) {		
			strm = new TextStream(file);

			// write smiles
			strm.writeln(smiles)
			strm.writeln('##############')

			// write spectrum
			for (i = 0; i < aPageItems.length; i++) {
				spec = new NMRSpectrum(aPageItems[i]);
				if (spec.isValid() && spec.dimCount === 1) {
					if (aReverse) {
						hz = spec.hz();
						dHz = spec.scaleWidth() / spec.count();
						pt = spec.count() - 1;
						dPt = -1;
						endPt = -1;
					} else {
						hz = spec.hz() + spec.scaleWidth();
						dHz = -spec.scaleWidth() / (spec.count()/10);
						pt = 0;
						dPt = 10;
						endPt = spec.count();
					}
					ppm = hz / spec.frequency();
					dPpm = dHz / spec.frequency();
					if (aReverse) {
						ppm += dPpm;
					}
					while (pt <= endPt) {					
						mapObj.hz = hz.toFixed(aDecimals);
						mapObj.ppm = ppm.toFixed(aDecimals);
						mapObj.pts = pt;
						mapObj.real = spec.real(pt).toFixed(aDecimals);
						mapObj.imag = spec.imag(pt).toFixed(aDecimals);
						strm.writeln(aFormat.formatMap(mapObj));
						hz += dHz;
						ppm += dPpm;
						pt += dPt;
						i+=1
					}
					break;
				}
			}
            strm.writeln('##############')

			// write multiplets
            var header = ',' + 'category' + ',' + 'rangeMax' + ',' + 'rangeMin' + ',' + 'nH' + ',' + 'centroid' + ',' + 'delta'
            strm.writeln(header)
            
            for (i=0; i<multiplets.count; i++){
                multiplet = multiplets.at(i)
                var line = i + ',' + multiplet.category + ',' + multiplet.rangeMax + ',' + multiplet.rangeMin + ',' + multiplet.nH + ',' + multiplet.centroid + ',' + multiplet.delta
                strm.writeln(line)
            }

		}
	} catch (e) {
		print("Exception found: {0}".format(e));
	} finally {
		file.close();
	}
};

