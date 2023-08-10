 /*globals TextStream, File, MoleculePlugin, Molecule, Application, NMRPredictor, molecule*/


function predict_13C(smiles, out_file) {

    "use strict";
    
    dw = Application.mainWindow.newDocument();
    
    mol = get_mol_from_smiles(dw, smiles);
    mol.normalizeNumbering();

    specId1H = NMRPredictor.predict(mol, "13C");
    predNMR = mol.nmrPrediction("13C", false);
                    
    // Save spectrum
   	var format = settings.value("Custom1DCsvConverter/CurrentFormat", "{ppm}{tab}{real}{tab}{imag}");
	save_spectrum_peaks(smiles, dw.curPage()['items'], out_file, format.replace(/\{tab\}/g, ","), 6, false, predNMR);   	
	
    // Destroy window
	dw.destroy();
}

function get_mol_from_smiles(aDocWin, aSMILES){
    var mol_id = molecule.importSMILES(aSMILES);
    mol = new Molecule(aDocWin.getItem(mol_id))
    return mol
}

function save_spectrum_peaks(smiles, aPageItems, aFilename, aFormat, aDecimals, aReverse, predNMR){
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
						dHz = -spec.scaleWidth() / (spec.count()/30);
						pt = 0;
						dPt = 30;
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

            		strm.writeln('##############')

					var header = ',' + 'atom (index)' + ',' + 'delta (ppm)'
					strm.writeln(header)
					
					for (i=0; i<predNMR['length']; i++){
						if (predNMR[i]['shift']['value'] == '-100000') {
							continue;
						}
												
						var line = i + ',' + predNMR[i]['atom']['index'] + ',' + predNMR[i]['shift']['value']
						
						strm.writeln(line)
					}
					break;
				}
			}
		}
	} catch (e) {
		print("Exception found: {0}".format(e));
	} finally {
		file.close();
	}
};
