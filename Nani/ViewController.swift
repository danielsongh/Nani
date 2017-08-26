import UIKit
import Vision
import AVFoundation


enum CameraError: Swift.Error {
    case captureSessionAlreadyRunning
    case captureSessionIsMissing
    case inputsAreInvalid
    case invalidOperation
    case noCamerasAvailable
    case unknown
}


class ViewController: UIViewController {
   
    var session: AVCaptureSession!
    var previewLayer: AVCaptureVideoPreviewLayer!
    
    let captureQueue = DispatchQueue(label: "captureQueue")
    
    
    var requests = [VNRequest]()
   

    @IBOutlet weak var previewView: UIView!
    @IBOutlet weak var predictionLabel: UILabel!
    
    
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        //predictionLabel.sizeToFit()
        
        setupVision()
        setupCamera()
        
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        
        previewLayer.frame = previewView.bounds
    }
    
    
    func setupCamera() {
        
        session = AVCaptureSession()
        
        do {

            guard let camera = AVCaptureDevice.default(for: .video) else {
                print("No camera is available")
                return
            }
            
            
            // Setup Camera Input
            let cameraInput = try AVCaptureDeviceInput(device: camera)
            
            if session.canAddInput(cameraInput) {
                session.addInput(cameraInput)
            }
            
           // Setup Camera Output
            let videoOutput = AVCaptureVideoDataOutput()

            videoOutput.setSampleBufferDelegate(self, queue: captureQueue)
            videoOutput.alwaysDiscardsLateVideoFrames = true
            videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            
            if session.canAddOutput(videoOutput){
                session.addOutput(videoOutput)
            }
            
            // Setup Preview Layer
            previewLayer = AVCaptureVideoPreviewLayer(session: session)
            previewLayer.videoGravity = .resizeAspectFill
            
            //previewView.layer.insertSublayer(previewLayer, at: 0)
            previewView.layer.addSublayer(previewLayer)
            
            // make sure we are in portrait mode
            let conn = videoOutput.connection(with: .video)
            conn?.videoOrientation = .portrait
            
            session.sessionPreset = .high

            // Start the session
            session.startRunning()
            
        } catch let error as NSError {
            print("error setting up \(error), \(error.userInfo)")
        }
    
    }
    
    func setupVision() {
        let classificationRequest: VNCoreMLRequest = {
            // Load the ML model through its generated class and create a Vision request for it.
            do {
                let model = try VNCoreMLModel(for: Resnet50().model)
                return VNCoreMLRequest(model: model, completionHandler: self.handleClassifications)
            } catch {
                fatalError("can't load Vision ML model: \(error)")
            }
        }()
        
        classificationRequest.imageCropAndScaleOption = .centerCrop
        
        requests = [classificationRequest]
    }
    
    
    func handleClassifications(request: VNRequest, error: Error?) {
        guard let observations = request.results as? [VNClassificationObservation]
            else { fatalError("unexpected result type from VNCoreMLRequest") }
        guard let best = observations.first
            else { fatalError("can't get best result") }

    
        DispatchQueue.main.async {
            self.predictionLabel.text = best.identifier
        }
    }
}


extension ViewController: AVCaptureVideoDataOutputSampleBufferDelegate{
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else{
            return
        }
        
        connection.videoOrientation = .portrait
        
        var requestOptions:[VNImageOption: Any] = [:]
        
        if let cameraIntrinsicData = CMGetAttachment(sampleBuffer, kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix, nil) {
            requestOptions = [.cameraIntrinsics: cameraIntrinsicData]
        }
        
        // for orientation see kCGImagePropertyOrientation
        let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .upMirrored, options: requestOptions)
        do {
            try imageRequestHandler.perform(requests)
        } catch {
            print(error)
        }
    }
}

